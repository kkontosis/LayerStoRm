// ExpertRidge (P11) engine-side inference — see expert_ridge.h.
// Semantics transcribed 1:1 from tools/elb_train/stagecprime_gpu.py
// (CPrimeState) and stagecprime_attn.py (CtxEncoder, frozen).

#include "prefetch/expert_ridge.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <limits>
#include <numeric>
#include <sstream>

#include <nlohmann/json.hpp>

#include "model/weight_loader/safetensors_reader.h"

namespace layerstorm::prefetch {

namespace {
constexpr double kRlsDecay = 0.999;
constexpr double kRlsLambda = 1.0;
constexpr int kRlsRefresh = 64;
constexpr double kCalDecay = 0.999;
constexpr float kRecencyCap = 512.0f;
constexpr float kPairDecay = 0.99f;
constexpr int kCtxWindow = 8;
constexpr int kRecurHorizon = 16;
const double kD2 = std::pow(0.5, 1.0 / 2.0);
const double kD8 = std::pow(0.5, 1.0 / 8.0);

// float8 e4m3 (fn) -> f32: 256-entry table built once. Matches
// torch.float8_e4m3fn semantics (bias 7, no inf, 0x7f/0xff = nan).
float e4m3_to_f32(uint8_t b) {
    const uint32_t sign = (b & 0x80u) ? 0x80000000u : 0u;
    const uint32_t exp = (b >> 3) & 0xF;
    const uint32_t man = b & 0x7;
    float out;
    if (exp == 0xF && man == 0x7) {
        out = std::numeric_limits<float>::quiet_NaN();
    } else if (exp == 0) {
        out = std::ldexp(static_cast<float>(man), -9);  // subnormal
    } else {
        out = std::ldexp(1.0f + man / 8.0f,
                         static_cast<int>(exp) - 7);
    }
    uint32_t bits;
    std::memcpy(&bits, &out, 4);
    bits |= sign;
    std::memcpy(&out, &bits, 4);
    return out;
}

// (A + lambda I) w = b, Gaussian elimination with partial pivoting.
// n <= F+1 (~19-25): trivially fast; f64 like torch.linalg.solve.
void solve_ridge(const double* A, const double* b, double* w, int n,
                 std::vector<double>& scratch) {
    scratch.assign(static_cast<size_t>(n) * (n + 1), 0.0);
    double* M = scratch.data();
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c)
            M[r * (n + 1) + c] = A[r * n + c] + (r == c ? kRlsLambda : 0.0);
        M[r * (n + 1) + n] = b[r];
    }
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(M[r * (n + 1) + col]) >
                std::fabs(M[piv * (n + 1) + col]))
                piv = r;
        if (piv != col)
            for (int c = col; c <= n; ++c)
                std::swap(M[col * (n + 1) + c], M[piv * (n + 1) + c]);
        const double d = M[col * (n + 1) + col];
        for (int r = col + 1; r < n; ++r) {
            const double f = M[r * (n + 1) + col] / d;
            if (f == 0.0) continue;
            for (int c = col; c <= n; ++c)
                M[r * (n + 1) + c] -= f * M[col * (n + 1) + c];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        double acc = M[r * (n + 1) + n];
        for (int c = r + 1; c < n; ++c) acc -= M[r * (n + 1) + c] * w[c];
        w[r] = acc / M[r * (n + 1) + r];
    }
}
}  // namespace

// ── init ─────────────────────────────────────────────────────────────────────

bool ExpertRidge::init_synthetic(int j, int e, int k, int memo_buckets,
                                 int memo2_buckets, int xsame_lag,
                                 int pool_m,
                                 const std::string& feat_names,
                                 const std::vector<float>& pca, int v,
                                 int d_tok, int ctx_d, int ctx_d_out,
                                 std::string* err) {
    j_ = j; e_ = e; k_ = k;
    memo_b_ = memo_buckets; memo2_b_ = memo2_buckets;
    xsame_lag_ = xsame_lag; pool_m_ = pool_m;
    v_ = v; d_tok_ = d_tok; ctx_d_ = ctx_d; ctx_dout_ = ctx_d_out;
    pca_ = pca;
    feat_names_.clear();
    std::stringstream ss(feat_names);
    std::string item;
    while (std::getline(ss, item, ',')) feat_names_.push_back(item);
    f_ = static_cast<int>(feat_names_.size());
    fa_ = f_ + 1;
    if (f_ < 12) { if (err) *err = "feature list too short"; return false; }
    has_memo_ = has_memo2_ = has_ctx_ = has_xsame_ = false;
    has_xu_ = has_xd_ = false;
    head_mask_.assign(fa_, 1.0);
    extra_cols_.assign(f_, 0);   // 0=unknown extra (zeros)
    for (int i = 0; i < f_; ++i) {
        const auto& n = feat_names_[i];
        if (n == "memo") { has_memo_ = true; extra_cols_[i] = 3; }
        if (n == "memo2") { has_memo2_ = true; extra_cols_[i] = 4; }
        if (n == "ctx") { has_ctx_ = true; head_mask_[i] = 0.0;
                          extra_cols_[i] = 5; }
        if (n == "xsame") { has_xsame_ = true; head_mask_[i] = 0.0;
                            extra_cols_[i] = 6; }
        if (n == "xprev_up") { has_xu_ = true; extra_cols_[i] = 1; }
        if (n == "xprev_dn") { has_xd_ = true; extra_cols_[i] = 2; }
    }
    const size_t jee = static_cast<size_t>(j_) * e_ * e_;
    trans_.assign(jee, 0.f);
    if (has_xu_) xu_.assign(jee, 0.f);
    if (has_xd_) xd_.assign(jee, 0.f);
    if (has_xsame_) xsame_.assign(jee, 0.f);
    if (has_memo_)
        memo_.assign(static_cast<size_t>(j_) * memo_b_ * e_, 0.f);
    if (has_memo2_)
        memo2_.assign(static_cast<size_t>(j_) * memo2_b_ * e_, 0.f);
    const size_t jaa = static_cast<size_t>(j_) * fa_ * fa_;
    const size_t ja = static_cast<size_t>(j_) * fa_;
    A_.assign(jaa, 0.0); Ao_.assign(jaa, 0.0); Asub_.assign(jaa, 0.0);
    b_.assign(ja, 0.0); bo_.assign(ja, 0.0); brec_.assign(ja, 0.0);
    bman_.assign(ja, 0.0);
    wr_.assign(ja, 0.0); wo_.assign(ja, 0.0); wrec_.assign(ja, 0.0);
    wman_.assign(ja, 0.0);
    calH_.assign(static_cast<size_t>(j_) * (e_ - k_), 0.f);
    calP_.assign(static_cast<size_t>(j_) * k_, 0.f);
    cal_n_ = 0.0;
    since_ = 0; t_idx_ = 0;
    rring_.clear(); rring_t_.clear();
    // scratch
    x_.assign(static_cast<size_t>(j_) * e_ * f_, 0.f);
    ctx_now_.assign(static_cast<size_t>(j_) * e_, 0.f);
    sc_open_.assign(static_cast<size_t>(j_) * e_, 0.0);
    sc_rec_.assign(static_cast<size_t>(j_) * e_, 0.0);
    sc_man_.assign(static_cast<size_t>(j_) * e_, 0.0);
    sc_open_f_.assign(static_cast<size_t>(j_) * e_, 0.f);
    sc_rec_f_.assign(static_cast<size_t>(j_) * e_, 0.f);
    sc_man_f_.assign(static_cast<size_t>(j_) * e_, 0.f);
    tp_.assign(static_cast<size_t>(j_) * e_, 0.f);
    trank_.assign(static_cast<size_t>(j_) * e_, 0.f);
    order_.assign(e_, 0);
    pool_ids_.assign(static_cast<size_t>(j_) * (pool_m_ - k_), 0);
    cal_curve_.assign(j_, 0.f);
    tmp_e_.assign(e_, 0.f);
    tmp_d_.assign(std::max({ctx_d_, d_tok_,
                            j_ * std::max(ctx_dout_, 1)}), 0.f);
    xa_row_.assign(fa_, 0.0);
    if (has_ctx_) {
        ctx_q_.assign(static_cast<size_t>(ctx_d_) * d_tok_, 0.f);
        ctx_k_ = ctx_q_; ctx_v_ = ctx_q_;
        ctx_pos_.assign(static_cast<size_t>(kCtxWindow) * ctx_d_, 0.f);
        ctx_projw_.assign(static_cast<size_t>(j_) * ctx_dout_ * ctx_d_,
                          0.f);
        ctx_projb_.assign(static_cast<size_t>(j_) * ctx_dout_, 0.f);
        ctx_expout_.assign(static_cast<size_t>(j_) * e_ * ctx_dout_,
                           0.f);
        ctx_bias_.assign(j_, 0.f);
    }
    begin_sequence();
    return true;
}

void ExpertRidge::set_ctx_params(
    std::vector<float> q, std::vector<float> k, std::vector<float> v,
    std::vector<float> pos, std::vector<float> proj_w,
    std::vector<float> proj_b, std::vector<float> exp_out,
    std::vector<float> bias) {
    ctx_q_ = std::move(q); ctx_k_ = std::move(k);
    ctx_v_ = std::move(v); ctx_pos_ = std::move(pos);
    ctx_projw_ = std::move(proj_w); ctx_projb_ = std::move(proj_b);
    ctx_expout_ = std::move(exp_out); ctx_bias_ = std::move(bias);
}

void ExpertRidge::begin_sequence() {
    const size_t je = static_cast<size_t>(j_) * e_;
    prev8_mask_.assign(je, 0.f);
    t16_.assign(je, 0.f); t64_.assign(je, 0.f);
    ema2_.assign(je, 0.f); ema8_.assign(je, 0.f);
    g_prev_.assign(je, 0.f); g_prev2_.assign(je, 0.f);
    cut_prev_.assign(j_, 0.f);
    prev8_.clear();
    last_seen_.assign(je, -1);
    n2_ = n8_ = 0.0;
    ring16_.clear(); ring64_.clear();
    ctx_ring_.clear();
    rring_.clear(); rring_t_.clear();   // per-seq (engine resets)
    t_idx_ = 0;
}

// ── ctx encoder (frozen forward) ────────────────────────────────────────────

void ExpertRidge::ctx_forward(int32_t tok, float* out) {
    const size_t je = static_cast<size_t>(j_) * e_;
    if (ctx_ring_.empty()) { std::fill(out, out + je, 0.f); return; }
    const int W = static_cast<int>(ctx_ring_.size());
    const float* cur = &pca_[static_cast<size_t>(tok) * d_tok_];
    std::vector<float>& q = tmp_d_;                  // [ctx_d]
    std::vector<float> att(W), cq(ctx_d_, 0.f);
    for (int d = 0; d < ctx_d_; ++d) {
        double acc = 0;
        const float* w = &ctx_q_[static_cast<size_t>(d) * d_tok_];
        for (int t = 0; t < d_tok_; ++t) acc += w[t] * cur[t];
        q[d] = static_cast<float>(acc);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(ctx_d_));
    float amax = -std::numeric_limits<float>::infinity();
    std::vector<float> kv(static_cast<size_t>(W) * ctx_d_);
    for (int i = 0; i < W; ++i) {
        const float* wt =
            &pca_[static_cast<size_t>(ctx_ring_[i]) * d_tok_];
        double dot = 0;
        for (int d = 0; d < ctx_d_; ++d) {
            double acc = 0;
            const float* wk = &ctx_k_[static_cast<size_t>(d) * d_tok_];
            for (int t = 0; t < d_tok_; ++t) acc += wk[t] * wt[t];
            const float kd = static_cast<float>(acc) +
                             ctx_pos_[static_cast<size_t>(i) * ctx_d_ + d];
            dot += static_cast<double>(kd) * q[d];
            double av = 0;
            const float* wv = &ctx_v_[static_cast<size_t>(d) * d_tok_];
            for (int t = 0; t < d_tok_; ++t) av += wv[t] * wt[t];
            kv[static_cast<size_t>(i) * ctx_d_ + d] =
                static_cast<float>(av);
        }
        att[i] = static_cast<float>(dot) * scale;
        amax = std::max(amax, att[i]);
    }
    double asum = 0;
    for (int i = 0; i < W; ++i) {
        att[i] = std::exp(att[i] - amax);
        asum += att[i];
    }
    for (int i = 0; i < W; ++i) att[i] /= static_cast<float>(asum);
    for (int d = 0; d < ctx_d_; ++d) {
        double acc = 0;
        for (int i = 0; i < W; ++i)
            acc += att[i] * kv[static_cast<size_t>(i) * ctx_d_ + d];
        cq[d] = static_cast<float>(acc) + q[d];
    }
    // h = projw (cq) + projb -> [J, dout] ONCE, then the [J,E,dout]
    // einsum (matches the Python compute shape; the naive per-(j,e)
    // recompute was 48x more MACs)
    std::vector<float> hbuf(static_cast<size_t>(j_) * ctx_dout_);
    for (int jj = 0; jj < j_; ++jj)
        for (int d = 0; d < ctx_dout_; ++d) {
            const size_t hrow = static_cast<size_t>(jj) * ctx_dout_ + d;
            double h = ctx_projb_[hrow];
            const float* pw = &ctx_projw_[hrow * ctx_d_];
            for (int c = 0; c < ctx_d_; ++c) h += pw[c] * cq[c];
            hbuf[hrow] = static_cast<float>(h);
        }
    for (int jj = 0; jj < j_; ++jj) {
        const float* hj = &hbuf[static_cast<size_t>(jj) * ctx_dout_];
        for (int ee = 0; ee < e_; ++ee) {
            const float* eo = &ctx_expout_[(static_cast<size_t>(jj) *
                                            e_ + ee) * ctx_dout_];
            float acc = 0;
            for (int d = 0; d < ctx_dout_; ++d) acc += hj[d] * eo[d];
            out[static_cast<size_t>(jj) * e_ + ee] =
                acc + ctx_bias_[jj];
        }
    }
}

// ── feature pass ─────────────────────────────────────────────────────────────

void ExpertRidge::build_features(const float* sel, const int32_t* tops,
                                 int32_t bucket, int32_t bucket2,
                                 int32_t token_id) {
    const bool have_prev = !prev8_.empty();
    const float inf = std::numeric_limits<float>::infinity();
    // pair scores (trans), rank features
    for (int jj = 0; jj < j_; ++jj) {
        float* tp = &tp_[static_cast<size_t>(jj) * e_];
        std::fill(tp, tp + e_, 0.f);
        if (have_prev)
            for (int kk = 0; kk < k_; ++kk) {
                const int p = prev8_[static_cast<size_t>(jj) * k_ + kk];
                const float* row =
                    &trans_[(static_cast<size_t>(jj) * e_ + p) * e_];
                for (int ee = 0; ee < e_; ++ee) tp[ee] += row[ee];
            }
        for (int ee = 0; ee < e_; ++ee) tp[ee] *= trans_scale_;
        // stable argsort of -tpm ascending (invalid = +inf tpm first)
        std::iota(order_.begin(), order_.end(), 0);
        const float* t64r = &t64_[static_cast<size_t>(jj) * e_];
        auto keyf = [&](int ee) {
            return t64r[ee] > 0.f ? -tp[ee] : -inf;
        };
        std::stable_sort(order_.begin(), order_.end(),
                         [&](int a, int b) { return keyf(a) < keyf(b); });
        int n_inv = 0;
        for (int ee = 0; ee < e_; ++ee)
            if (!(t64r[ee] > 0.f)) ++n_inv;
        const float denom =
            std::max(1, (e_ - n_inv) - 1);
        float* tr = &trank_[static_cast<size_t>(jj) * e_];
        for (int r = 0; r < e_; ++r)
            tr[order_[r]] = (static_cast<float>(r) - n_inv) / denom;
    }
    if (has_ctx_) ctx_forward(token_id, ctx_now_.data());
    const float e2n = n2_ > 0 ? static_cast<float>(1.0 / n2_) : 1.f;
    const float e8n = n8_ > 0 ? static_cast<float>(1.0 / n8_) : 1.f;
    for (int jj = 0; jj < j_; ++jj) {
        const size_t jo = static_cast<size_t>(jj) * e_;
        const float layer_f =
            j_ > 1 ? static_cast<float>(jj) / (j_ - 1) : 0.f;
        // memo/memo2 normalized rows
        const float* mrow = nullptr;
        float mnorm = 0.f;
        if (has_memo_ && bucket >= 0) {
            mrow = &memo_[(static_cast<size_t>(jj) * memo_b_ + bucket) *
                          e_];
            double s = 0;
            for (int ee = 0; ee < e_; ++ee) s += mrow[ee];
            mnorm = static_cast<float>(1.0 / (1.0 + s));
        }
        const float* m2row = nullptr;
        float m2norm = 0.f;
        if (has_memo2_ && bucket2 >= 0) {
            m2row = &memo2_[(static_cast<size_t>(jj) * memo2_b_ +
                             bucket2) * e_];
            double s = 0;
            for (int ee = 0; ee < e_; ++ee) s += m2row[ee];
            m2norm = static_cast<float>(1.0 / (1.0 + s));
        }
        for (int ee = 0; ee < e_; ++ee) {
            float* xr = &x_[(jo + ee) * f_];
            const long ls = last_seen_[jo + ee];
            const float rec =
                ls >= 0 ? std::min(static_cast<float>(t_idx_ - ls),
                                   kRecencyCap)
                        : kRecencyCap;
            int c = 0;
            xr[c++] = prev8_mask_[jo + ee];
            xr[c++] = t16_[jo + ee];
            xr[c++] = t64_[jo + ee];
            xr[c++] = rec;
            xr[c++] = tp_[jo + ee];
            xr[c++] = trank_[jo + ee];
            xr[c++] = ema2_[jo + ee] * e2n;
            xr[c++] = ema8_[jo + ee] * e8n;
            xr[c++] = g_prev_[jo + ee];
            xr[c++] = g_prev_[jo + ee] - cut_prev_[jj];
            xr[c++] = g_prev_[jo + ee] - g_prev2_[jo + ee];
            xr[c++] = layer_f;
            for (int fi = 12; fi < f_; ++fi) {
                const int code = extra_cols_[fi];
                float v = 0.f;
                if (code == 1) {
                    if (have_prev && jj + 1 < j_) {
                        for (int kk = 0; kk < k_; ++kk) {
                            const int p =
                                prev8_[(static_cast<size_t>(jj) + 1) *
                                       k_ + kk];
                            v += xu_[(static_cast<size_t>(jj) * e_ + p) *
                                     e_ + ee];
                        }
                        v *= xu_scale_;
                    }
                } else if (code == 2) {
                    if (have_prev && jj >= 1) {
                        for (int kk = 0; kk < k_; ++kk) {
                            const int p =
                                prev8_[(static_cast<size_t>(jj) - 1) *
                                       k_ + kk];
                            v += xd_[(static_cast<size_t>(jj) * e_ + p) *
                                     e_ + ee];
                        }
                        v *= xd_scale_;
                    }
                } else if (code == 3) {
                    v = mrow ? mrow[ee] * mnorm : 0.f;
                } else if (code == 4) {
                    v = m2row ? m2row[ee] * m2norm : 0.f;
                } else if (code == 5) {
                    v = ctx_now_[jo + ee];
                } else if (code == 6) {
                    if (jj >= xsame_lag_) {
                        for (int kk = 0; kk < k_; ++kk) {
                            const int p =
                                tops[(static_cast<size_t>(jj) -
                                      xsame_lag_) * k_ + kk];
                            v += xsame_[(static_cast<size_t>(jj) * e_ +
                                         p) * e_ + ee];
                        }
                        v *= xsame_scale_;
                    }
                }
                xr[c++] = v;
            }
        }
    }
}

void ExpertRidge::score_heads() {
    for (int jj = 0; jj < j_; ++jj) {
        const double* wo = &wo_[static_cast<size_t>(jj) * fa_];
        const double* wr = &wrec_[static_cast<size_t>(jj) * fa_];
        const double* wm = &wman_[static_cast<size_t>(jj) * fa_];
        for (int ee = 0; ee < e_; ++ee) {
            const float* xr =
                &x_[(static_cast<size_t>(jj) * e_ + ee) * f_];
            double so = wo[f_], sr = wr[f_], sm = wm[f_];  // bias
            for (int fi = 0; fi < f_; ++fi) {
                const double xv = static_cast<double>(xr[fi]);
                so += xv * wo[fi];
                sr += xv * wr[fi];
                sm += xv * wm[fi];
            }
            const size_t o = static_cast<size_t>(jj) * e_ + ee;
            sc_open_[o] = so; sc_rec_[o] = sr; sc_man_[o] = sm;
            sc_open_f_[o] = static_cast<float>(so);
            sc_rec_f_[o] = static_cast<float>(sr);
            sc_man_f_[o] = static_cast<float>(sm);
        }
    }
}

// ── pi-hat ───────────────────────────────────────────────────────────────────

void ExpertRidge::cal_update(const float* sc_open,
                             const int32_t* tops) {
    const float neg = std::numeric_limits<float>::lowest();
    for (int jj = 0; jj < j_; ++jj) {
        const size_t jo = static_cast<size_t>(jj) * e_;
        std::iota(order_.begin(), order_.end(), 0);
        std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) {
            const float sa =
                prev8_mask_[jo + a] > 0 ? neg : sc_open[jo + a];
            const float sb =
                prev8_mask_[jo + b] > 0 ? neg : sc_open[jo + b];
            return sa > sb;
        });
        // yb membership for this layer
        for (int ee = 0; ee < e_; ++ee) tmp_e_[ee] = 0.f;
        for (int kk = 0; kk < k_; ++kk)
            tmp_e_[tops[static_cast<size_t>(jj) * k_ + kk]] = 1.f;
        float* Hr = &calH_[static_cast<size_t>(jj) * (e_ - k_)];
        for (int r = 0; r < e_ - k_; ++r)
            Hr[r] = static_cast<float>(Hr[r] * kCalDecay +
                                       tmp_e_[order_[r]] *
                                           (1.0 - kCalDecay));
        std::iota(order_.begin(), order_.end(), 0);
        std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) {
            const float sa =
                prev8_mask_[jo + a] > 0 ? sc_open[jo + a] : neg;
            const float sb =
                prev8_mask_[jo + b] > 0 ? sc_open[jo + b] : neg;
            return sa > sb;
        });
        float* Pr = &calP_[static_cast<size_t>(jj) * k_];
        for (int r = 0; r < k_; ++r)
            Pr[r] = static_cast<float>(Pr[r] * kCalDecay +
                                       tmp_e_[order_[r]] *
                                           (1.0 - kCalDecay));
    }
    cal_n_ = cal_n_ * kCalDecay + (1.0 - kCalDecay);
}

// ── updates ──────────────────────────────────────────────────────────────────

void ExpertRidge::update_state(const float* sel, const int32_t* tops,
                               const float* /*top_w*/, int32_t bucket,
                               int32_t bucket2, int32_t /*token_id*/) {
    const size_t je = static_cast<size_t>(j_) * e_;
    // ridge sufficient statistics (f64), from the SCORING-time X
    for (size_t i = 0; i < A_.size(); ++i) {
        A_[i] *= kRlsDecay; Ao_[i] *= kRlsDecay; Asub_[i] *= kRlsDecay;
    }
    for (size_t i = 0; i < b_.size(); ++i) {
        b_[i] *= kRlsDecay; bo_[i] *= kRlsDecay; brec_[i] *= kRlsDecay;
        bman_[i] *= kRlsDecay;
    }
    std::vector<float> xa_snap(je * fa_);
    for (int jj = 0; jj < j_; ++jj) {
        double* Aj = &A_[static_cast<size_t>(jj) * fa_ * fa_];
        double* Aoj = &Ao_[static_cast<size_t>(jj) * fa_ * fa_];
        double* Asj = &Asub_[static_cast<size_t>(jj) * fa_ * fa_];
        double* bj = &b_[static_cast<size_t>(jj) * fa_];
        double* boj = &bo_[static_cast<size_t>(jj) * fa_];
        for (int ee = 0; ee < e_; ++ee) {
            const size_t o = static_cast<size_t>(jj) * e_ + ee;
            const float* xr = &x_[o * f_];
            double* xa = xa_row_.data();
            for (int fi = 0; fi < f_; ++fi)
                xa[fi] = static_cast<double>(xr[fi]);
            xa[f_] = 1.0;
            float* snap = &xa_snap[o * fa_];
            for (int fi = 0; fi < fa_; ++fi)
                snap[fi] = static_cast<float>(xa[fi]);
            const bool valid = t64_[o] > 0.f;
            bool y = false;
            for (int kk = 0; kk < k_; ++kk)
                if (tops[static_cast<size_t>(jj) * k_ + kk] == ee)
                    y = true;
            for (int r = 0; r < fa_; ++r) {
                const double xr_r = xa[r];
                if (xr_r != 0.0) {
                    double* Ar = &Aoj[static_cast<size_t>(r) * fa_];
                    for (int c = 0; c < fa_; ++c) Ar[c] += xr_r * xa[c];
                    if (valid) {
                        double* Av = &Aj[static_cast<size_t>(r) * fa_];
                        for (int c = 0; c < fa_; ++c)
                            Av[c] += xr_r * xa[c];
                    }
                    const double xm_r = xr_r * head_mask_[r];
                    if (xm_r != 0.0) {
                        double* As = &Asj[static_cast<size_t>(r) * fa_];
                        for (int c = 0; c < fa_; ++c)
                            As[c] += xm_r * xa[c] * head_mask_[c];
                    }
                }
                if (y) {
                    boj[r] += xa[r];
                    if (valid) bj[r] += xa[r];
                }
            }
        }
    }
    rring_.push_back(std::move(xa_snap));
    rring_t_.push_back(t_idx_);
    // refresh solve — Python order: AFTER the A/b accumulation,
    // BEFORE the pair-table updates and this position's recur-label
    // resolution (the matured label lands post-solve)
    if (++since_ >= kRlsRefresh) {
        refresh_solve();
        since_ = 0;
    }
    // pair tables (xsame first — within-position; then trans/xprev)
    auto pair_update = [&](std::vector<float>& tbl, float& scale,
                           const int32_t* src, const int32_t* dst,
                           int lo, int hi) {
        scale *= kPairDecay;
        const float inc = 1.0f / scale;
        for (int jj = lo; jj < hi; ++jj) {
            // src row index offset: aligned slice, see callers
            const int32_t* s = src + static_cast<size_t>(jj - lo) * k_;
            const int32_t* d = dst + static_cast<size_t>(jj - lo) * k_;
            for (int a = 0; a < k_; ++a)
                for (int bidx = 0; bidx < k_; ++bidx)
                    tbl[(static_cast<size_t>(jj) * e_ + s[a]) * e_ +
                        d[bidx]] += inc;
        }
        if (scale < 1e-20f) {
            for (auto& t : tbl) t *= scale;
            scale = 1.0f;
        }
    };
    if (has_xsame_ && xsame_lag_ > 0)
        pair_update(xsame_, xsame_scale_, tops,
                    tops + static_cast<size_t>(xsame_lag_) * k_,
                    xsame_lag_, j_);
    if (!prev8_.empty()) {
        pair_update(trans_, trans_scale_, prev8_.data(), tops, 0, j_);
        if (has_xu_)
            pair_update(xu_, xu_scale_, prev8_.data() + k_, tops, 0,
                        j_ - 1);
        if (has_xd_)
            pair_update(xd_, xd_scale_, prev8_.data(),
                        tops + static_cast<size_t>(1) * k_, 1, j_);
    }
    // trailing windows
    auto ring_step = [&](std::vector<std::vector<int32_t>>& ring,
                         std::vector<float>& counts, size_t hor) {
        if (ring.size() >= hor) {
            const auto& old = ring.front();
            for (int jj = 0; jj < j_; ++jj)
                for (int kk = 0; kk < k_; ++kk)
                    counts[static_cast<size_t>(jj) * e_ +
                           old[static_cast<size_t>(jj) * k_ + kk]] -=
                        1.f;
            ring.erase(ring.begin());
        }
        ring.emplace_back(tops, tops + static_cast<size_t>(j_) * k_);
        for (int jj = 0; jj < j_; ++jj)
            for (int kk = 0; kk < k_; ++kk)
                counts[static_cast<size_t>(jj) * e_ +
                       tops[static_cast<size_t>(jj) * k_ + kk]] += 1.f;
    };
    ring_step(ring16_, t16_, 16);
    ring_step(ring64_, t64_, 64);
    // EMAs
    for (size_t i = 0; i < je; ++i) {
        ema2_[i] = static_cast<float>(ema2_[i] * kD2 +
                                      sel[i] * (1.0 - kD2));
        ema8_[i] = static_cast<float>(ema8_[i] * kD8 +
                                      sel[i] * (1.0 - kD8));
    }
    n2_ = n2_ * kD2 + (1.0 - kD2);
    n8_ = n8_ * kD8 + (1.0 - kD8);
    // memo counts
    if (has_memo_ && bucket >= 0)
        for (int jj = 0; jj < j_; ++jj)
            for (int kk = 0; kk < k_; ++kk)
                memo_[(static_cast<size_t>(jj) * memo_b_ + bucket) *
                          e_ +
                      tops[static_cast<size_t>(jj) * k_ + kk]] += 1.f;
    if (has_memo2_ && bucket2 >= 0)
        for (int jj = 0; jj < j_; ++jj)
            for (int kk = 0; kk < k_; ++kk)
                memo2_[(static_cast<size_t>(jj) * memo2_b_ + bucket2) *
                           e_ +
                       tops[static_cast<size_t>(jj) * k_ + kk]] += 1.f;
    // prev / score snapshots
    prev8_.assign(tops, tops + static_cast<size_t>(j_) * k_);
    std::fill(prev8_mask_.begin(), prev8_mask_.end(), 0.f);
    for (int jj = 0; jj < j_; ++jj)
        for (int kk = 0; kk < k_; ++kk)
            prev8_mask_[static_cast<size_t>(jj) * e_ +
                        tops[static_cast<size_t>(jj) * k_ + kk]] = 1.f;
    g_prev2_ = g_prev_;
    g_prev_.assign(sel, sel + je);
    for (int jj = 0; jj < j_; ++jj) {
        // k-th largest of sel row
        std::vector<float>& t = tmp_e_;
        std::copy(sel + static_cast<size_t>(jj) * e_,
                  sel + static_cast<size_t>(jj + 1) * e_, t.begin());
        std::nth_element(t.begin(), t.begin() + (k_ - 1),
                         t.begin() + e_, std::greater<float>());
        cut_prev_[jj] = t[k_ - 1];
    }
    for (int jj = 0; jj < j_; ++jj)
        for (int kk = 0; kk < k_; ++kk)
            last_seen_[static_cast<size_t>(jj) * e_ +
                       tops[static_cast<size_t>(jj) * k_ + kk]] = t_idx_;
    ++t_idx_;
    // recur-16 delayed labels (masked-column contributions)
    if (rring_.size() > kRecurHorizon) {
        const size_t idx = rring_.size() - (kRecurHorizon + 1);
        const auto& xa0 = rring_[idx];
        const long t0 = rring_t_[idx];
        for (int jj = 0; jj < j_; ++jj) {
            double* br = &brec_[static_cast<size_t>(jj) * fa_];
            for (int ee = 0; ee < e_; ++ee) {
                const size_t o = static_cast<size_t>(jj) * e_ + ee;
                if (last_seen_[o] > t0) {
                    const float* xr = &xa0[o * fa_];
                    for (int fi = 0; fi < fa_; ++fi)
                        br[fi] += static_cast<double>(xr[fi]) *
                                  head_mask_[fi];
                }
            }
        }
    }
    while (rring_.size() > kRecurHorizon) {
        rring_.erase(rring_.begin());
        rring_t_.erase(rring_t_.begin());
    }
}

void ExpertRidge::refresh_solve() {
    std::vector<double> scratch;
    for (int jj = 0; jj < j_; ++jj) {
        const size_t ao = static_cast<size_t>(jj) * fa_ * fa_;
        const size_t bo = static_cast<size_t>(jj) * fa_;
        solve_ridge(&A_[ao], &b_[bo], &wr_[bo], fa_, scratch);
        solve_ridge(&Ao_[ao], &bo_[bo], &wo_[bo], fa_, scratch);
        solve_ridge(&Asub_[ao], &brec_[bo], &wrec_[bo], fa_, scratch);
        solve_ridge(&Asub_[ao], &bman_[bo], &wman_[bo], fa_, scratch);
    }
}

// ── step ─────────────────────────────────────────────────────────────────────

const ExpertRidgeAdvice& ExpertRidge::advise(const float* sel,
                                             const int32_t* tops,
                                             int32_t token_bucket,
                                             int32_t bigram_bucket,
                                             int32_t token_id) {
    if (t_idx_ >= 1) {
        build_features(sel, tops, token_bucket, bigram_bucket,
                       token_id);
        score_heads();
        // pool ranking (prev-exempt) + pi-hat curve, PRE-update
        const float neg = std::numeric_limits<float>::lowest();
        for (int jj = 0; jj < j_; ++jj) {
            const size_t jo = static_cast<size_t>(jj) * e_;
            std::iota(order_.begin(), order_.end(), 0);
            std::stable_sort(
                order_.begin(), order_.end(), [&](int a, int b) {
                    const float sa = prev8_mask_[jo + a] > 0
                                         ? neg
                                         : sc_open_f_[jo + a];
                    const float sb = prev8_mask_[jo + b] > 0
                                         ? neg
                                         : sc_open_f_[jo + b];
                    return sa > sb;
                });
            for (int r = 0; r < pool_m_ - k_; ++r)
                pool_ids_[static_cast<size_t>(jj) * (pool_m_ - k_) +
                          r] = order_[r];
            // C_j(pool_m): prev slots + top-(pool_m-K) open ranks
            float c = 0.f;
            if (cal_n_ > 0) {
                const float* Pr = &calP_[static_cast<size_t>(jj) * k_];
                for (int r = 0; r < k_; ++r) c += Pr[r];
                const float* Hr =
                    &calH_[static_cast<size_t>(jj) * (e_ - k_)];
                const int m = std::min(pool_m_ - k_, e_ - k_);
                for (int r = 0; r < m; ++r) c += Hr[r];
                c = static_cast<float>(c / cal_n_);
            }
            cal_curve_[jj] = c;
        }
    } else {
        std::fill(x_.begin(), x_.end(), 0.f);
        std::fill(sc_open_.begin(), sc_open_.end(), 0.0);
        std::fill(sc_rec_.begin(), sc_rec_.end(), 0.0);
        std::fill(sc_man_.begin(), sc_man_.end(), 0.0);
        std::fill(sc_open_f_.begin(), sc_open_f_.end(), 0.f);
        std::fill(sc_rec_f_.begin(), sc_rec_f_.end(), 0.f);
        std::fill(sc_man_f_.begin(), sc_man_f_.end(), 0.f);
        std::fill(pool_ids_.begin(), pool_ids_.end(), 0);
        std::fill(cal_curve_.begin(), cal_curve_.end(), 0.f);
    }
    advice_.open_scores = sc_open_f_.data();
    advice_.evict_scores = sc_rec_f_.data();
    advice_.manifest_scores = sc_man_f_.data();
    advice_.pool_ids = pool_ids_.data();
    advice_.cal_curve = cal_curve_.data();
    advice_.j = j_; advice_.e = e_; advice_.k = k_;
    advice_.pool_m = pool_m_;
    return advice_;
}

const ExpertRidgeAdvice& ExpertRidge::step(const float* sel,
                                           const int32_t* tops,
                                           const float* top_w,
                                           int32_t token_bucket,
                                           int32_t bigram_bucket,
                                           int32_t token_id) {
    advise(sel, tops, token_bucket, bigram_bucket, token_id);
    if (t_idx_ >= 1)
        cal_update(sc_open_f_.data(), tops);
    // ctx token ring maintenance (frozen encoder; ring after predict)
    if (has_ctx_) {
        ctx_ring_.push_back(token_id);
        if (static_cast<int>(ctx_ring_.size()) > kCtxWindow)
            ctx_ring_.erase(ctx_ring_.begin());
    }
    update_state(sel, tops, top_w, token_bucket, bigram_bucket,
                 token_id);
    return advice_;
}

// ── bundle loading ───────────────────────────────────────────────────────────

namespace {
// Load an e4m3 table "<base>.fp8" + "<base>.scale" ([J] per-layer
// amax scales) and dequantize into dst (row-major [J, E, E]).
bool load_fp8_table(const layerstorm::model::SafetensorsReader& rd,
                    const std::string& base, std::vector<float>& dst,
                    size_t jee, int J) {
    const layerstorm::model::TensorEntry* eb = nullptr;
    const layerstorm::model::TensorEntry* es = nullptr;
    for (const auto& e : rd.entries()) {
        if (e.name == base + ".fp8") eb = &e;
        if (e.name == base + ".scale") es = &e;
    }
    if (!eb || !es || eb->data_size_bytes != jee) return false;
    auto bits = rd.tensor_data(*eb);
    auto ssp = rd.tensor_data(*es);
    std::vector<float> scale(J);
    if (ssp.size() != static_cast<size_t>(J) * 4) return false;
    std::memcpy(scale.data(), ssp.data(), ssp.size());
    static float lut[256];
    static bool lut_init = false;
    if (!lut_init) {
        for (int i = 0; i < 256; ++i)
            lut[i] = e4m3_to_f32(static_cast<uint8_t>(i));
        lut_init = true;
    }
    dst.resize(jee);
    const size_t per_j = jee / J;
    const auto* pb = reinterpret_cast<const uint8_t*>(bits.data());
    for (int jj = 0; jj < J; ++jj) {
        const float sc = scale[jj];
        const size_t o = static_cast<size_t>(jj) * per_j;
        for (size_t i = 0; i < per_j; ++i)
            dst[o + i] = lut[pb[o + i]] * sc;
    }
    return true;
}
}  // namespace

bool ExpertRidge::load(const std::string& model_dir, std::string* err) {
    namespace fs = std::filesystem;
    using layerstorm::model::SafetensorsReader;
    try {
        const fs::path dir(model_dir);
        std::ifstream spec_f(dir / "feature_spec.json");
        if (!spec_f) {
            if (err) *err = "feature_spec.json missing";
            return false;
        }
        nlohmann::json spec = nlohmann::json::parse(spec_f);
        std::string feats;
        for (const auto& f : spec.at("features_in_order")) {
            const std::string s = f.get<std::string>();
            if (s == "bias") continue;
            if (!feats.empty()) feats += ",";
            feats += s;
        }
        int lag = 0;
        if (spec.contains("pipelined_protocol"))
            lag = spec["pipelined_protocol"].value("xsame_lag", 0);
        auto rd = SafetensorsReader::open(dir / "model.safetensors",
                                          /*use_mmap=*/true);
        std::optional<SafetensorsReader> fp8_reader;
        if (fs::exists(dir / "model.fp8.safetensors"))
            fp8_reader = SafetensorsReader::open(
                dir / "model.fp8.safetensors", /*use_mmap=*/true);
        auto find = [&](const std::string& name)
            -> const layerstorm::model::TensorEntry* {
            for (const auto& e : rd.entries())
                if (e.name == name) return &e;
            return nullptr;
        };
        const auto* wo = find("ridge.w_open");
        if (!wo) { if (err) *err = "ridge.w_open missing"; return false; }
        const int J = static_cast<int>(wo->shape[0]);
        const int FA = static_cast<int>(wo->shape[1]);
        const auto* tr = find("trans.fp8") ? nullptr : find("trans");
        const auto* xs = find("xsame");
        const int E = 256, K = 8;   // engine constants (feature_spec
        //                              scoring contract)
        // memo buckets from metadata-free stamp: dense rebuild is
        // deliberately NOT done here (sparse stamps stay sparse);
        // v1 loads heads + pair tables + ctx + calibration.
        std::vector<float> pca;     // PCA table
        const auto* pe = find("embed_ctx32.features");
        int V = 0, DT = 32;
        if (pe) {
            V = static_cast<int>(pe->shape[0]);
            DT = static_cast<int>(pe->shape[1]);
        }
        pca.resize(static_cast<size_t>(V) * DT);
        // (decode below)
        if (!init_synthetic(J, E, K, /*memo_b=*/1, /*memo2_b=*/1, lag,
                            /*pool_m=*/32, feats, pca, V, DT,
                            /*ctx_d=*/48, /*ctx_d_out=*/32, err))
            return false;
        has_memo_ = has_memo2_ = false;  // sparse stamps: v1 gathers
        //  are zeros until the stamp consumer lands (documented)
        auto load_f32 = [&](const char* name, std::vector<float>& dst,
                            size_t want) {
            const auto* e = find(name);
            if (!e || static_cast<size_t>(e->data_size_bytes) !=
                          want * 4)
                return false;
            auto sp = rd.tensor_data(*e);
            dst.resize(want);
            std::memcpy(dst.data(), sp.data(), want * 4);
            return true;
        };
        auto load_f64 = [&](const char* name, std::vector<double>& dst,
                            size_t want) {
            const auto* e = find(name);
            if (!e) return false;
            auto sp = rd.tensor_data(*e);
            dst.resize(want);
            std::memcpy(dst.data(), sp.data(), want * 8);
            return true;
        };
        const size_t ja = static_cast<size_t>(J) * FA;
        if (!load_f64("ridge.w_open", wo_, ja) ||
            !load_f64("ridge.w_recur", wrec_, ja) ||
            !load_f64("ridge.w_manifest", wman_, ja)) {
            if (err) *err = "ridge heads missing";
            return false;
        }
        const size_t jee = static_cast<size_t>(J) * E * E;
        // pair tables: prefer the CERTIFIED fp8 bundle (e4m3 bits x
        // per-layer f32 scale) when present next to the f32 dir
        auto load_pair = [&](const char* f32_name,
                             const char* fp8_base,
                             std::vector<float>& dst) {
            if (fp8_reader &&
                load_fp8_table(*fp8_reader, fp8_base, dst, jee, J))
                return true;
            return load_f32(f32_name, dst, jee);
        };
        load_pair("trans", "trans", trans_);
        load_pair("xsame", "xsame", xsame_);
        load_pair("xprev.xu", "xprev.xu", xu_);
        load_pair("xprev.xd", "xprev.xd", xd_);
        (void)tr; (void)xs;
        if (pe) {
            // PCA features ride F16 in the bundle
            auto sp = rd.tensor_data(*pe);
            const uint16_t* h =
                reinterpret_cast<const uint16_t*>(sp.data());
            pca_.resize(static_cast<size_t>(V) * DT);
            for (size_t i = 0; i < pca_.size(); ++i) {
                const uint16_t x = h[i];
                const uint32_t sign = (x & 0x8000u) << 16;
                uint32_t exp = (x >> 10) & 0x1F;
                uint32_t man = x & 0x3FF;
                uint32_t f;
                if (exp == 0) {
                    if (man == 0) { f = sign; }
                    else {
                        exp = 127 - 15 + 1;
                        while (!(man & 0x400)) { man <<= 1; --exp; }
                        man &= 0x3FF;
                        f = sign | (exp << 23) | (man << 13);
                    }
                } else if (exp == 31) {
                    f = sign | 0x7F800000u | (man << 13);
                } else {
                    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
                }
                std::memcpy(&pca_[i], &f, 4);
            }
        }
        load_f32("ctx_encoder.q.weight", ctx_q_,
                 static_cast<size_t>(ctx_d_) * DT);
        load_f32("ctx_encoder.k.weight", ctx_k_,
                 static_cast<size_t>(ctx_d_) * DT);
        load_f32("ctx_encoder.v.weight", ctx_v_,
                 static_cast<size_t>(ctx_d_) * DT);
        load_f32("ctx_encoder.pos.weight", ctx_pos_,
                 static_cast<size_t>(kCtxWindow) * ctx_d_);
        load_f32("ctx_encoder.proj.weight", ctx_projw_,
                 static_cast<size_t>(J) * ctx_dout_ * ctx_d_);
        load_f32("ctx_encoder.proj.bias", ctx_projb_,
                 static_cast<size_t>(J) * ctx_dout_);
        load_f32("ctx_encoder.exp_out", ctx_expout_,
                 static_cast<size_t>(J) * E * ctx_dout_);
        load_f32("ctx_encoder.bias", ctx_bias_, J);
        load_f32("calibration.H", calH_,
                 static_cast<size_t>(J) * (E - K));
        load_f32("calibration.P", calP_, static_cast<size_t>(J) * K);
        std::vector<double> n1;
        if (load_f64("calibration.n_cal", n1, 1)) cal_n_ = n1[0];
        if (cal_n_ <= 0) {
            // 0-d scalars ride the safetensors __metadata__ (the
            // bundle writer's contract) — parse the header directly
            std::ifstream bf(dir / "model.safetensors",
                             std::ios::binary);
            uint64_t hlen = 0;
            bf.read(reinterpret_cast<char*>(&hlen), 8);
            std::string hjson(hlen, '\0');
            bf.read(hjson.data(), static_cast<std::streamsize>(hlen));
            auto hj = nlohmann::json::parse(hjson);
            if (hj.contains("__metadata__")) {
                const auto& md = hj["__metadata__"];
                if (md.contains("calibration.n_cal"))
                    cal_n_ = std::stod(
                        md["calibration.n_cal"].get<std::string>());
            }
        }
        // the bundle stores the STATIONARY form H/n_cal — restore
        // the engine-state raw decayed tables
        if (cal_n_ > 0) {
            for (auto& x : calH_) x = static_cast<float>(x * cal_n_);
            for (auto& x : calP_) x = static_cast<float>(x * cal_n_);
        }
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = ex.what();
        return false;
    }
}

}  // namespace layerstorm::prefetch
