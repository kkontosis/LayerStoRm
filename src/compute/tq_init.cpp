// TurboQuant initialization: codebook loading and rotation matrix generation.
// See tq_init.h for public API and spec/TURBOQUANT_DETAILS.md §14-§15.

#include "compute/tq_init.h"
#include "core/attention_device.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace layerstorm::compute {

// ── Codebook loading ────────────────────────────────────────────────────────

TqCodebook load_codebook(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("TQ codebook not found: " + path);
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("TQ codebook JSON parse error: " + std::string(e.what()));
    }

    TqCodebook cb;

    // Required fields
    if (!j.contains("d") || !j.contains("bits") || !j.contains("centroids") ||
        !j.contains("boundaries")) {
        throw std::runtime_error("TQ codebook missing required fields (d, bits, centroids, boundaries)");
    }

    cb.d    = j["d"].get<int>();
    cb.bits = j["bits"].get<int>();
    cb.n_clusters = 1 << cb.bits;

    cb.centroids  = j["centroids"].get<std::vector<float>>();
    cb.boundaries = j["boundaries"].get<std::vector<float>>();

    // Validate shapes
    if (static_cast<int>(cb.centroids.size()) != cb.n_clusters) {
        throw std::runtime_error(
            "TQ codebook: expected " + std::to_string(cb.n_clusters) +
            " centroids, got " + std::to_string(cb.centroids.size()));
    }
    if (static_cast<int>(cb.boundaries.size()) != cb.n_clusters + 1) {
        throw std::runtime_error(
            "TQ codebook: expected " + std::to_string(cb.n_clusters + 1) +
            " boundaries, got " + std::to_string(cb.boundaries.size()));
    }

    // Validate sorted
    if (!std::is_sorted(cb.centroids.begin(), cb.centroids.end())) {
        throw std::runtime_error("TQ codebook: centroids not sorted");
    }
    if (!std::is_sorted(cb.boundaries.begin(), cb.boundaries.end())) {
        throw std::runtime_error("TQ codebook: boundaries not sorted");
    }

    // Extract interior boundaries (what kernels use for searchsorted)
    // boundaries[1 .. n_clusters-1] — omit the -1.0 and +1.0 endpoints
    cb.interior_boundaries.resize(cb.n_clusters - 1);
    for (int i = 0; i < cb.n_clusters - 1; ++i) {
        cb.interior_boundaries[i] = cb.boundaries[i + 1];
    }

    return cb;
}

// ── Householder QR decomposition ────────────────────────────────────────────
//
// Computes Q, R from a d×d Gaussian matrix G such that G = Q*R.
// After sign-fixing (Q[:,i] *= sign(R[i,i])), Q is a proper rotation
// matrix with det(Q) = +1 (INV-TQ-6).
//
// Computed in double precision for numerical stability, then cast to float32.

namespace {

/// Explicit Box-Muller transform for cross-compiler determinism.
/// std::normal_distribution is implementation-defined, so we avoid it.
void fill_gaussian(double* out, int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    constexpr double two_pi = 2.0 * 3.14159265358979323846;

    int i = 0;
    for (; i + 1 < n; i += 2) {
        double u1 = uniform(rng);
        double u2 = uniform(rng);
        // Guard against u1 == 0 (log(0) is undefined)
        if (u1 < 1e-300) u1 = 1e-300;
        double r = std::sqrt(-2.0 * std::log(u1));
        out[i]     = r * std::cos(two_pi * u2);
        out[i + 1] = r * std::sin(two_pi * u2);
    }
    if (i < n) {
        double u1 = uniform(rng);
        double u2 = uniform(rng);
        if (u1 < 1e-300) u1 = 1e-300;
        out[i] = std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
    }
}

/// In-place Householder QR on a d×d row-major matrix.
/// On entry:  A = random Gaussian (will be overwritten to R).
/// On exit:   Q = orthogonal factor, A = upper triangular R.
void householder_qr(double* A, double* Q, int d) {
    // Initialize Q = I
    for (int i = 0; i < d * d; ++i) Q[i] = 0.0;
    for (int i = 0; i < d; ++i) Q[i * d + i] = 1.0;

    std::vector<double> v(d);

    for (int k = 0; k < d; ++k) {
        // Extract column k of A below diagonal: x = A[k:d, k]
        double norm_sq = 0.0;
        for (int i = k; i < d; ++i) {
            double val = A[i * d + k];
            norm_sq += val * val;
        }
        double norm_x = std::sqrt(norm_sq);

        if (norm_x < 1e-300) continue;  // zero column, skip

        // sigma = -sign(A[k,k]) * ||x||
        double sigma = (A[k * d + k] >= 0.0) ? -norm_x : norm_x;

        // v = x; v[0] -= sigma  =>  v[0] = A[k,k] - sigma
        for (int i = k; i < d; ++i) v[i - k] = A[i * d + k];
        v[0] -= sigma;

        // Normalise: tau = 2 / (v^T v)
        double v_norm_sq = 0.0;
        int len = d - k;
        for (int i = 0; i < len; ++i) v_norm_sq += v[i] * v[i];
        if (v_norm_sq < 1e-300) continue;
        double tau = 2.0 / v_norm_sq;

        // Apply to A:  A[k:d, k:d] -= tau * v * (v^T * A[k:d, k:d])
        for (int j = k; j < d; ++j) {
            double dot = 0.0;
            for (int i = 0; i < len; ++i) {
                dot += v[i] * A[(k + i) * d + j];
            }
            double scale = tau * dot;
            for (int i = 0; i < len; ++i) {
                A[(k + i) * d + j] -= scale * v[i];
            }
        }

        // Apply to Q:  Q[:, k:d] -= tau * (Q[:, k:d] * v) * v^T
        for (int i = 0; i < d; ++i) {
            double dot = 0.0;
            for (int j = 0; j < len; ++j) {
                dot += Q[i * d + (k + j)] * v[j];
            }
            double scale = tau * dot;
            for (int j = 0; j < len; ++j) {
                Q[i * d + (k + j)] -= scale * v[j];
            }
        }

        // Set diagonal explicitly
        A[k * d + k] = -sigma;
    }
}

}  // namespace

void generate_rotation_matrix_cpu(float* Pi, int d, int seed) {
    assert(d > 0);
    const int n = d * d;

    // Generate Gaussian matrix in double precision
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::vector<double> G(n);
    fill_gaussian(G.data(), n, rng);

    // QR decomposition (double precision)
    std::vector<double> Q(n);
    householder_qr(G.data(), Q.data(), d);
    // G is now R (upper triangular), Q is orthogonal factor

    // Sign-fixing: Q[:,i] *= sign(R[i,i])
    // This makes R unique (non-negative diagonal) but doesn't guarantee det=+1.
    // Householder QR applies d reflections → det(Q_raw) = (-1)^d.
    // Each column flip for negative R[i,i] toggles the sign → det = (-1)^(d + num_flips).
    // Final correction: if det=-1, negate column 0 to get det=+1 (INV-TQ-6).
    int num_flips = 0;
    for (int i = 0; i < d; ++i) {
        double diag = G[i * d + i];  // R[i,i]
        if (diag < 0.0) {
            ++num_flips;
            for (int j = 0; j < d; ++j) {
                Q[j * d + i] = -Q[j * d + i];
            }
        }
    }

    // Ensure proper rotation: det(Q) = (-1)^(d + num_flips).
    // If odd, flip one more column to get det=+1.
    if ((d + num_flips) % 2 != 0) {
        for (int j = 0; j < d; ++j) {
            Q[j * d + 0] = -Q[j * d + 0];
        }
    }

    // Cast to float32 (row-major)
    for (int i = 0; i < n; ++i) {
        Pi[i] = static_cast<float>(Q[i]);
    }
}

// ── Orthogonality / determinant verification ────────────────────────────────

float verify_orthogonality(const float* Q, int d) {
    // Compute max |Q^T Q - I|
    float max_err = 0.0f;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < d; ++k) {
                dot += Q[k * d + i] * Q[k * d + j];  // Q^T[i,k] * Q[k,j]
            }
            float expected = (i == j) ? 1.0f : 0.0f;
            float err = std::abs(dot - expected);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

float compute_determinant(const float* Q, int d) {
    // LU decomposition in double precision
    std::vector<double> A(d * d);
    for (int i = 0; i < d * d; ++i) A[i] = Q[i];

    double det = 1.0;
    for (int k = 0; k < d; ++k) {
        // Partial pivoting
        int max_row = k;
        double max_val = std::abs(A[k * d + k]);
        for (int i = k + 1; i < d; ++i) {
            double val = std::abs(A[i * d + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }
        if (max_row != k) {
            for (int j = 0; j < d; ++j) std::swap(A[k * d + j], A[max_row * d + j]);
            det = -det;  // row swap flips sign
        }

        double pivot = A[k * d + k];
        if (std::abs(pivot) < 1e-300) return 0.0f;
        det *= pivot;

        for (int i = k + 1; i < d; ++i) {
            double factor = A[i * d + k] / pivot;
            for (int j = k + 1; j < d; ++j) {
                A[i * d + j] -= factor * A[k * d + j];
            }
        }
    }
    return static_cast<float>(det);
}

// ── TqResources ─────────────────────────────────────────────────────────────

const float* TqResources::device_Pi(int layer_idx) const {
    assert(layer_idx >= 0 && layer_idx < static_cast<int>(rotations.size()));
    return rotations[layer_idx].d_Pi;
}

const float* TqResources::device_Pi_t(int layer_idx) const {
    assert(layer_idx >= 0 && layer_idx < static_cast<int>(rotations.size()));
    return rotations[layer_idx].d_Pi_t;
}

// ── Init / destroy ──────────────────────────────────────────────────────────

std::shared_ptr<const TqRotationHostData> precompute_tq_rotations(
        int d, int num_layers) {
    auto out = std::make_shared<TqRotationHostData>();
    out->d = d;
    out->Pi.resize(static_cast<size_t>(num_layers));
    out->Pi_t.resize(static_cast<size_t>(num_layers));
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned nthreads =
        std::min<unsigned>(hw, static_cast<unsigned>(std::max(1, num_layers)));
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    std::atomic<int> next{0};
    for (unsigned t = 0; t < nthreads; ++t) {
        workers.emplace_back([&] {
            for (int layer = next.fetch_add(1); layer < num_layers;
                 layer = next.fetch_add(1)) {
                auto& pi = out->Pi[static_cast<size_t>(layer)];
                pi.resize(static_cast<size_t>(d) * d);
                // Same seed law as the per-rank path (INV-TQ-4).
                generate_rotation_matrix_cpu(pi.data(), d, 42 + layer * 7);
                auto& pi_t = out->Pi_t[static_cast<size_t>(layer)];
                pi_t.resize(static_cast<size_t>(d) * d);
                for (int r = 0; r < d; ++r)
                    for (int col = 0; col < d; ++col)
                        pi_t[static_cast<size_t>(col) * d + r] =
                            pi[static_cast<size_t>(r) * d + col];
            }
        });
    }
    for (auto& w : workers) w.join();
    spdlog::info("TQ init: precomputed {} shared rotation matrices "
                 "(d={}, {} thread(s))", num_layers, d, nthreads);
    return out;
}

std::unique_ptr<TqResources> init_tq_resources(
        const TqInitOptions& opts,
        const TqRotationHostData* shared_rotations) {
    if (opts.d_c <= 0) {
        throw std::runtime_error("TQ init: d_c must be positive");
    }
    if (opts.num_layers <= 0) {
        throw std::runtime_error("TQ init: num_layers must be positive");
    }
    if (!opts.attention_device) {
        throw std::runtime_error("TQ init: device_backend must not be null");
    }

    auto res = std::make_unique<TqResources>();

    // ── Codebook ──

    std::string codebook_path = opts.codebook_dir + "/codebook_d" +
                                std::to_string(opts.d_c) + "_b" +
                                std::to_string(opts.bits) + ".json";

    spdlog::info("TQ init: loading codebook from {}", codebook_path);
    res->codebook = load_codebook(codebook_path);

    // Validate dimension matches
    if (res->codebook.d != opts.d_c) {
        throw std::runtime_error(
            "TQ codebook dimension mismatch: file has d=" +
            std::to_string(res->codebook.d) + " but config has d_c=" +
            std::to_string(opts.d_c));
    }

    // Upload centroids to device
    size_t centroids_bytes = res->codebook.n_clusters * sizeof(float);
    res->codebook.d_centroids = static_cast<float*>(
        opts.attention_device->device_alloc(centroids_bytes));
    if (!res->codebook.d_centroids) {
        throw std::runtime_error("TQ init: failed to allocate device centroids");
    }
    opts.attention_device->memcpy_h2d(res->codebook.d_centroids,
                                      res->codebook.centroids.data(), centroids_bytes);

    // Upload interior boundaries to device
    size_t boundaries_bytes = res->codebook.interior_boundaries.size() * sizeof(float);
    res->codebook.d_boundaries = static_cast<float*>(
        opts.attention_device->device_alloc(boundaries_bytes));
    if (!res->codebook.d_boundaries) {
        throw std::runtime_error("TQ init: failed to allocate device boundaries");
    }
    opts.attention_device->memcpy_h2d(res->codebook.d_boundaries,
                                      res->codebook.interior_boundaries.data(),
                                      boundaries_bytes);

    spdlog::info("TQ init: codebook uploaded ({} centroids, {} interior boundaries)",
                 res->codebook.n_clusters,
                 res->codebook.interior_boundaries.size());

    // ── Rotation matrices ──

    const int d = opts.d_c;
    const size_t pi_bytes = static_cast<size_t>(d) * d * sizeof(float);

    res->rotations.resize(opts.num_layers);
    std::vector<float> pi_host(d * d);

    // Shared host data (startup lever): the Π contents are rank-independent
    // (seed law has no rank), so a precomputed set skips the ~0.3 s/layer
    // Householder QR here — only the per-rank device uploads remain.
    const bool use_shared =
        shared_rotations && shared_rotations->d == d &&
        static_cast<int>(shared_rotations->Pi.size()) == opts.num_layers &&
        static_cast<int>(shared_rotations->Pi_t.size()) == opts.num_layers;
    if (shared_rotations && !use_shared)
        spdlog::warn("TQ init: shared rotations mismatch (d={} layers={}) — "
                     "regenerating per rank", shared_rotations->d,
                     shared_rotations->Pi.size());

    for (int layer = 0; layer < opts.num_layers; ++layer) {
        auto& rot = res->rotations[layer];
        rot.d         = d;
        rot.layer_idx = layer;
        rot.seed      = 42 + layer * 7;  // INV-TQ-4

        const float* pi_src;
        const float* pi_t_src;
        std::vector<float> pi_t_host;
        if (use_shared) {
            pi_src   = shared_rotations->Pi[layer].data();
            pi_t_src = shared_rotations->Pi_t[layer].data();
            rot.Pi   = shared_rotations->Pi[layer];  // host copy (tests)
        } else {
            // Generate on CPU
            generate_rotation_matrix_cpu(pi_host.data(), d, rot.seed);
            rot.Pi = std::vector<float>(pi_host.begin(), pi_host.end());
            // Transposed copy: makes the inverse rotation (out·Pi) a
            // row-access GEMV over Pi^T rows — same kernel as forward.
            pi_t_host.resize(static_cast<size_t>(d) * d);
            for (int r = 0; r < d; ++r)
                for (int col = 0; col < d; ++col)
                    pi_t_host[static_cast<size_t>(col) * d + r] =
                        pi_host[static_cast<size_t>(r) * d + col];
            pi_src   = pi_host.data();
            pi_t_src = pi_t_host.data();
        }

        // Upload to device (always per-rank, INV-TQ-PERRANK)
        rot.d_Pi = static_cast<float*>(opts.attention_device->device_alloc(pi_bytes));
        if (!rot.d_Pi) {
            throw std::runtime_error(
                "TQ init: failed to allocate device Π for layer " + std::to_string(layer));
        }
        opts.attention_device->memcpy_h2d(rot.d_Pi, pi_src, pi_bytes);

        rot.d_Pi_t = static_cast<float*>(
            opts.attention_device->device_alloc(pi_bytes));
        if (!rot.d_Pi_t) {
            throw std::runtime_error(
                "TQ init: failed to allocate device Pi^T for layer "
                + std::to_string(layer));
        }
        opts.attention_device->memcpy_h2d(rot.d_Pi_t, pi_t_src, pi_bytes);
    }

    spdlog::info("TQ init: generated {} rotation matrices (d={}, {:.1f} MB total)",
                 opts.num_layers, d,
                 static_cast<double>(opts.num_layers) * pi_bytes / (1024.0 * 1024.0));

    return res;
}

void destroy_tq_resources(TqResources& res, AttentionDevice& device) {
    // Free codebook device memory
    if (res.codebook.d_centroids) {
        device.device_free(res.codebook.d_centroids);
        res.codebook.d_centroids = nullptr;
    }
    if (res.codebook.d_boundaries) {
        device.device_free(res.codebook.d_boundaries);
        res.codebook.d_boundaries = nullptr;
    }

    // Free rotation matrices
    for (auto& rot : res.rotations) {
        if (rot.d_Pi_t) {
            device.device_free(rot.d_Pi_t);
            rot.d_Pi_t = nullptr;
        }
        if (rot.d_Pi) {
            device.device_free(rot.d_Pi);
            rot.d_Pi = nullptr;
        }
    }
}

}  // namespace layerstorm::compute
