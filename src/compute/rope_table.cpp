#include "compute/rope_table.h"

#include <algorithm>
#include <cmath>

namespace layerstorm::compute {

namespace {

// dim index below which frequencies complete fewer than num_rotations over max_seq_len
// (DeepSeek reference find_correction_dim).
double find_correction_dim(double num_rotations, int dim, double base,
                           double max_seq_len) {
    return dim * std::log(max_seq_len / (num_rotations * 2.0 * M_PI))
         / (2.0 * std::log(base));
}

}  // namespace

std::vector<float> build_rope_cos_sin_table(
    int max_pos, int d_rope, double rope_theta,
    const std::optional<config::RopeScalingConfig>& scaling) {

    const int half = d_rope / 2;
    std::vector<double> freqs(half);
    for (int i = 0; i < half; ++i)
        freqs[i] = 1.0 / std::pow(rope_theta,
                                  static_cast<double>(2 * i) / d_rope);

    // YaRN frequency correction (DeepSeek reference precompute_freqs_cis):
    // freqs = freqs/factor·(1−smooth) + freqs·smooth, smooth = 1 − ramp(low, high).
    if (scaling && scaling->type == config::RopeScalingType::yarn
        && scaling->factor && scaling->original_max_position_embeddings
        && max_pos > static_cast<int>(*scaling->original_max_position_embeddings)) {
        const double factor = *scaling->factor;
        const double beta_fast = scaling->beta_fast.value_or(32.0);
        const double beta_slow = scaling->beta_slow.value_or(1.0);
        const double orig = *scaling->original_max_position_embeddings;

        double low_d = std::floor(find_correction_dim(beta_fast, d_rope, rope_theta, orig));
        double high_d = std::ceil(find_correction_dim(beta_slow, d_rope, rope_theta, orig));
        double low = std::max(low_d, 0.0);
        double high = std::min(high_d, static_cast<double>(d_rope - 1));
        if (low == high) high += 0.001;

        for (int i = 0; i < half; ++i) {
            const double ramp = std::clamp((i - low) / (high - low), 0.0, 1.0);
            const double smooth = 1.0 - ramp;
            freqs[i] = freqs[i] / factor * (1.0 - smooth) + freqs[i] * smooth;
        }
    }

    std::vector<float> table(static_cast<size_t>(max_pos) * d_rope);
    for (int p = 0; p < max_pos; ++p) {
        float* row = table.data() + static_cast<size_t>(p) * d_rope;
        for (int i = 0; i < half; ++i) {
            const double ang = static_cast<double>(p) * freqs[i];
            row[i] = static_cast<float>(std::cos(ang));
            row[half + i] = static_cast<float>(std::sin(ang));
        }
    }
    return table;
}

V4RopeTables build_v4_rope_tables(
    int max_pos, int d_rope, double rope_theta, double compress_rope_theta,
    const std::optional<config::RopeScalingConfig>& scaling) {
    V4RopeTables t;
    // Uncompressed layers: base theta, NO yarn (freq_scale 1, ext_factor 0 —
    // deepseek4.cpp:819-821 zeroes every yarn parameter when ratio == 0).
    t.base = build_rope_cos_sin_table(max_pos, d_rope, rope_theta,
                                      /*scaling=*/std::nullopt);
    // Compressed layers: compress theta WITH the yarn frequency correction.
    t.compress = build_rope_cos_sin_table(max_pos, d_rope, compress_rope_theta,
                                          scaling);
    return t;
}

float rope_softmax_scale(
    int qk_nope_head_dim, int qk_rope_head_dim, int max_seq_len,
    const std::optional<config::RopeScalingConfig>& scaling) {

    double scale = 1.0 / std::sqrt(static_cast<double>(qk_nope_head_dim
                                                       + qk_rope_head_dim));
    if (scaling && scaling->type == config::RopeScalingType::yarn
        && scaling->factor && scaling->original_max_position_embeddings
        && max_seq_len > static_cast<int>(*scaling->original_max_position_embeddings)) {
        const double m_cfg = scaling->mscale_all_dim
            ? *scaling->mscale_all_dim
            : scaling->mscale.value_or(1.0);
        const double mscale = 0.1 * m_cfg * std::log(*scaling->factor) + 1.0;
        scale *= mscale * mscale;
    }
    return static_cast<float>(scale);
}

}  // namespace layerstorm::compute
