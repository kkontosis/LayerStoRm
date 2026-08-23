// TurboQuant initialization: codebook loading and rotation matrix generation.
//
// At engine startup (when attention_backend == turboquant_mla), loads the
// Lloyd-Max codebook from JSON and generates per-layer orthogonal rotation
// matrices Π via Householder QR.  Both are uploaded to device memory as
// float* pointers consumed by TQ CUDA kernels (tq_fused_k_append,
// tq_q_rotate, tq_decode, tq_v_rotate_back).
//
// See spec/TURBOQUANT_DETAILS.md §14 (codebook) and §15 (rotation).

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace layerstorm::compute {

class AttentionDevice;

// ── Codebook ────────────────────────────────────────────────────────────────

/// Lloyd-Max codebook for TQ_MSE 4-bit quantization.
/// Loaded from JSON, uploaded to device memory.
struct TqCodebook {
    int d          = 0;   // dimension (kv_lora_rank: 512 for V3.2, 448 for MODEL1)
    int bits       = 4;
    int n_clusters = 16;  // 2^bits

    // Host copies (validation, tests)
    std::vector<float> centroids;            // [n_clusters] sorted ascending
    std::vector<float> boundaries;           // [n_clusters + 1] sorted, -1.0 … +1.0
    std::vector<float> interior_boundaries;  // [n_clusters - 1] = boundaries[1..n_clusters-1]

    // Device pointers (owned; freed by destroy_tq_resources)
    float* d_centroids  = nullptr;  // [n_clusters] on device
    float* d_boundaries = nullptr;  // [n_clusters - 1] interior boundaries on device
};

// ── Rotation matrix ─────────────────────────────────────────────────────────

/// Per-layer rotation matrix Π (d × d, float32).
/// Generated deterministically from seed = 42 + layer_idx * 7 (INV-TQ-4).
struct TqRotationMatrix {
    int d         = 0;
    int layer_idx = 0;
    int seed      = 0;

    // Host copy (tests)
    std::vector<float> Pi;  // [d * d] row-major

    // Device pointer (owned; freed by destroy_tq_resources)
    float* d_Pi = nullptr;  // [d, d] on device
    float* d_Pi_t = nullptr;  // [d, d] TRANSPOSED on device — rows are Pi
                              // columns, so the inverse rotation becomes the
                              // same row-access GEMV as the forward one
                              // (TD-TQ-SPARSE-DECODE §12l rotate optimization)
};

// ── Resources aggregate ─────────────────────────────────────────────────────

/// All TQ resources for a model instance.
/// Created by init_tq_resources(), freed by destroy_tq_resources().
struct TqResources {
    TqCodebook codebook;
    std::vector<TqRotationMatrix> rotations;  // [num_layers]

    // Convenience accessors for kernel params
    const float* device_centroids() const { return codebook.d_centroids; }
    const float* device_boundaries() const { return codebook.d_boundaries; }
    const float* device_Pi(int layer_idx) const;
    const float* device_Pi_t(int layer_idx) const;
    int d_c() const { return codebook.d; }
    int num_layers() const { return static_cast<int>(rotations.size()); }
};

// ── Init options ────────────────────────────────────────────────────────────

struct TqInitOptions {
    int d_c         = 0;   // kv_lora_rank (512 or 448)
    int bits        = 4;   // always 4
    int num_layers  = 0;   // num_hidden_layers
    std::string codebook_dir;    // path to directory containing codebook_d{d}_b{bits}.json
    AttentionDevice* attention_device = nullptr;  // for device_alloc / device_free / memcpy_h2d
};

// ── Shared host-side rotations (startup-time lever) ─────────────────────────

/// Host-side per-layer rotation data (Π + Π^T), RANK-INDEPENDENT: the INV-TQ-4
/// seeds (42 + layer*7) contain no rank, so every rank uploads identical
/// bytes. Householder QR of a 512×512 costs ~0.3 s/layer single-threaded
/// (~23 s per rank for 79 layers — measured as the dominant rank-init cost);
/// precompute ONCE, layer-parallel, and share across ranks: each rank's
/// init_tq_resources then only allocs + uploads (~78 MB ×2, sub-second).
struct TqRotationHostData {
    int d = 0;
    std::vector<std::vector<float>> Pi;    // [num_layers][d*d] row-major
    std::vector<std::vector<float>> Pi_t;  // [num_layers][d*d] transposed
};

/// Generate all layers' Π/Π^T on the host, parallelized across layers.
/// Bit-identical to the per-rank generation it replaces (same seeds, same
/// generate_rotation_matrix_cpu per layer; layers are independent).
std::shared_ptr<const TqRotationHostData> precompute_tq_rotations(
    int d, int num_layers);

// ── Lifecycle ───────────────────────────────────────────────────────────────

/// Initialise all TQ resources: load codebook, generate per-layer Π,
/// upload to device.  Throws on failure (missing codebook, alloc failure).
/// `shared_rotations` (optional): use precomputed host Π/Π^T instead of
/// regenerating per rank (must match d_c/num_layers — a mismatch falls back
/// to per-rank generation with a warning). Device uploads stay per-rank
/// (INV-TQ-PERRANK).
std::unique_ptr<TqResources> init_tq_resources(
    const TqInitOptions& opts,
    const TqRotationHostData* shared_rotations = nullptr);

/// Free all device memory in a TqResources.
void destroy_tq_resources(TqResources& res, AttentionDevice& device);

// ── CPU helpers (exposed for testing) ───────────────────────────────────────

/// Load codebook from a JSON file.  Throws if file not found or malformed.
TqCodebook load_codebook(const std::string& path);

/// Generate rotation matrix on CPU via Householder QR in double precision.
/// Deterministic from seed.  Result written to Pi (row-major, d*d floats).
void generate_rotation_matrix_cpu(float* Pi, int d, int seed);

/// Max |Q^T Q - I| entry.  For testing orthogonality.
float verify_orthogonality(const float* Q, int d);

/// Determinant of a d×d float32 matrix via LU decomposition.  For testing.
float compute_determinant(const float* Q, int d);

}  // namespace layerstorm::compute
