// Tests for TurboQuant initialization: codebook loading and rotation matrix generation.

#include <gtest/gtest.h>

#include "compute/tq_init.h"
#include "core/null_attention_device.h"
#include "config/config_parser.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <vector>

using namespace layerstorm::compute;
using layerstorm::config::GpuRef;
using layerstorm::config::GpuType;

// ── Helpers ─────────────────────────────────────────────────────────────────

#ifdef LAYERSTORM_SOURCE_DIR
static std::string codebook_dir() {
    return std::string(LAYERSTORM_SOURCE_DIR) + "/config/tq_codebooks";
}
#endif

static std::unique_ptr<AttentionDevice> make_null_backend() {
    return make_null_attention_device(GpuRef{0, 0, GpuType::rtx5090});
}

// ── Suite 1: Codebook loading ───────────────────────────────────────────────

#ifdef LAYERSTORM_SOURCE_DIR

TEST(TqCodebookLoad, LoadsD512B4) {
    auto cb = load_codebook(codebook_dir() + "/codebook_d512_b4.json");

    EXPECT_EQ(cb.d, 512);
    EXPECT_EQ(cb.bits, 4);
    EXPECT_EQ(cb.n_clusters, 16);
    ASSERT_EQ(cb.centroids.size(), 16u);
    ASSERT_EQ(cb.boundaries.size(), 17u);
    ASSERT_EQ(cb.interior_boundaries.size(), 15u);

    // Sorted
    EXPECT_TRUE(std::is_sorted(cb.centroids.begin(), cb.centroids.end()));
    EXPECT_TRUE(std::is_sorted(cb.boundaries.begin(), cb.boundaries.end()));

    // Boundaries span [-1, 1]
    EXPECT_FLOAT_EQ(cb.boundaries.front(), -1.0f);
    EXPECT_FLOAT_EQ(cb.boundaries.back(), 1.0f);

    // Symmetric centroids: c[i] ≈ -c[n-1-i]
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(cb.centroids[i], -cb.centroids[15 - i], 1e-6f)
            << "Centroid symmetry failed at index " << i;
    }

    // Interior boundaries exclude endpoints
    EXPECT_GT(cb.interior_boundaries[0], -1.0f);
    EXPECT_LT(cb.interior_boundaries[14], 1.0f);
}

TEST(TqCodebookLoad, LoadsD448B4) {
    auto cb = load_codebook(codebook_dir() + "/codebook_d448_b4.json");

    EXPECT_EQ(cb.d, 448);
    EXPECT_EQ(cb.bits, 4);
    EXPECT_EQ(cb.n_clusters, 16);
    ASSERT_EQ(cb.centroids.size(), 16u);
    ASSERT_EQ(cb.boundaries.size(), 17u);
    ASSERT_EQ(cb.interior_boundaries.size(), 15u);

    EXPECT_TRUE(std::is_sorted(cb.centroids.begin(), cb.centroids.end()));
    EXPECT_FLOAT_EQ(cb.boundaries.front(), -1.0f);
    EXPECT_FLOAT_EQ(cb.boundaries.back(), 1.0f);
}

#endif  // LAYERSTORM_SOURCE_DIR

TEST(TqCodebookLoad, RejectsNonexistent) {
    EXPECT_THROW(load_codebook("/nonexistent/path/codebook.json"), std::runtime_error);
}

TEST(TqCodebookLoad, RejectsMalformed) {
    // Write a temp file with invalid JSON
    const char* tmp = "/tmp/tq_test_malformed.json";
    {
        std::ofstream f(tmp);
        f << "{ not valid json }}}";
    }
    EXPECT_THROW(load_codebook(tmp), std::runtime_error);
    std::remove(tmp);
}

TEST(TqCodebookLoad, RejectsMissingFields) {
    const char* tmp = "/tmp/tq_test_missing.json";
    {
        std::ofstream f(tmp);
        f << R"({"d": 512, "bits": 4})";  // missing centroids, boundaries
    }
    EXPECT_THROW(load_codebook(tmp), std::runtime_error);
    std::remove(tmp);
}

TEST(TqCodebookLoad, RejectsWrongCentroidCount) {
    const char* tmp = "/tmp/tq_test_wrong_count.json";
    {
        std::ofstream f(tmp);
        f << R"({
            "d": 64, "bits": 4,
            "centroids": [1.0, 2.0, 3.0],
            "boundaries": [-1.0, 0.0, 1.0]
        })";
    }
    EXPECT_THROW(load_codebook(tmp), std::runtime_error);
    std::remove(tmp);
}

// ── Suite 2: Rotation matrix generation ─────────────────────────────────────

TEST(TqRotationMatrix, Deterministic) {
    const int d = 64;
    const int seed = 42;
    std::vector<float> pi1(d * d), pi2(d * d);

    generate_rotation_matrix_cpu(pi1.data(), d, seed);
    generate_rotation_matrix_cpu(pi2.data(), d, seed);

    for (int i = 0; i < d * d; ++i) {
        EXPECT_EQ(pi1[i], pi2[i]) << "Mismatch at index " << i;
    }
}

TEST(TqRotationMatrix, DifferentSeeds) {
    const int d = 64;
    std::vector<float> pi1(d * d), pi2(d * d);

    generate_rotation_matrix_cpu(pi1.data(), d, 42);
    generate_rotation_matrix_cpu(pi2.data(), d, 49);

    // They should differ (extremely unlikely to be identical)
    bool any_diff = false;
    for (int i = 0; i < d * d; ++i) {
        if (pi1[i] != pi2[i]) { any_diff = true; break; }
    }
    EXPECT_TRUE(any_diff);
}

TEST(TqRotationMatrix, OrthogonalD64) {
    const int d = 64;
    std::vector<float> pi(d * d);
    generate_rotation_matrix_cpu(pi.data(), d, 42);

    float max_err = verify_orthogonality(pi.data(), d);
    EXPECT_LT(max_err, 1e-5f) << "Q^T Q deviates from I by " << max_err;
}

TEST(TqRotationMatrix, OrthogonalD128) {
    const int d = 128;
    std::vector<float> pi(d * d);
    generate_rotation_matrix_cpu(pi.data(), d, 42);

    float max_err = verify_orthogonality(pi.data(), d);
    EXPECT_LT(max_err, 1e-5f) << "Q^T Q deviates from I by " << max_err;
}

TEST(TqRotationMatrix, OrthogonalD448) {
    const int d = 448;
    std::vector<float> pi(d * d);
    generate_rotation_matrix_cpu(pi.data(), d, 42);

    float max_err = verify_orthogonality(pi.data(), d);
    EXPECT_LT(max_err, 1e-4f) << "Q^T Q deviates from I by " << max_err;
}

TEST(TqRotationMatrix, OrthogonalD512) {
    const int d = 512;
    std::vector<float> pi(d * d);
    generate_rotation_matrix_cpu(pi.data(), d, 42);

    float max_err = verify_orthogonality(pi.data(), d);
    EXPECT_LT(max_err, 1e-4f) << "Q^T Q deviates from I by " << max_err;
}

TEST(TqRotationMatrix, PositiveDeterminantD64) {
    const int d = 64;
    std::vector<float> pi(d * d);
    generate_rotation_matrix_cpu(pi.data(), d, 42);

    float det = compute_determinant(pi.data(), d);
    EXPECT_NEAR(det, 1.0f, 1e-3f) << "det(Q) should be +1 (INV-TQ-6), got " << det;
}

TEST(TqRotationMatrix, PositiveDeterminantD512) {
    const int d = 512;
    std::vector<float> pi(d * d);
    generate_rotation_matrix_cpu(pi.data(), d, 42);

    float det = compute_determinant(pi.data(), d);
    EXPECT_NEAR(det, 1.0f, 0.1f) << "det(Q) should be +1 (INV-TQ-6), got " << det;
}

TEST(TqRotationMatrix, SeedConvention) {
    // INV-TQ-4: seed = 42 + layer_idx * 7
    const int d = 16;
    std::vector<float> pi_layer0(d * d), pi_layer3(d * d);

    generate_rotation_matrix_cpu(pi_layer0.data(), d, 42 + 0 * 7);  // seed=42
    generate_rotation_matrix_cpu(pi_layer3.data(), d, 42 + 3 * 7);  // seed=63

    // Verify layer 3 seed is 63 (same result as explicit seed=63)
    std::vector<float> pi_seed63(d * d);
    generate_rotation_matrix_cpu(pi_seed63.data(), d, 63);
    for (int i = 0; i < d * d; ++i) {
        EXPECT_EQ(pi_layer3[i], pi_seed63[i]);
    }

    // Different from layer 0
    bool any_diff = false;
    for (int i = 0; i < d * d; ++i) {
        if (pi_layer0[i] != pi_layer3[i]) { any_diff = true; break; }
    }
    EXPECT_TRUE(any_diff);
}

// ── Suite 3: TqResources lifecycle (NullAttentionDevice) ────────────────────

#ifdef LAYERSTORM_SOURCE_DIR

TEST(TqResourcesLifecycle, AllocAndFree) {
    auto backend = make_null_backend();
    TqInitOptions opts;
    opts.d_c           = 512;
    opts.bits          = 4;
    opts.num_layers    = 3;
    opts.codebook_dir  = codebook_dir();
    opts.attention_device = backend.get();


    auto res = init_tq_resources(opts);
    ASSERT_NE(res, nullptr);

    // Cleanup should not crash
    destroy_tq_resources(*res, *backend);
}

TEST(TqResourcesLifecycle, DevicePointersNonNull) {
    auto backend = make_null_backend();
    TqInitOptions opts;
    opts.d_c           = 512;
    opts.bits          = 4;
    opts.num_layers    = 2;
    opts.codebook_dir  = codebook_dir();
    opts.attention_device = backend.get();


    auto res = init_tq_resources(opts);

    EXPECT_NE(res->device_centroids(), nullptr);
    EXPECT_NE(res->device_boundaries(), nullptr);
    EXPECT_NE(res->device_Pi(0), nullptr);
    EXPECT_NE(res->device_Pi(1), nullptr);

    destroy_tq_resources(*res, *backend);
}

TEST(TqResourcesLifecycle, CodebookShared) {
    auto backend = make_null_backend();
    TqInitOptions opts;
    opts.d_c           = 448;
    opts.bits          = 4;
    opts.num_layers    = 3;
    opts.codebook_dir  = codebook_dir();
    opts.attention_device = backend.get();


    auto res = init_tq_resources(opts);

    // All layers use the same codebook device pointers
    const float* c = res->device_centroids();
    const float* b = res->device_boundaries();
    EXPECT_NE(c, nullptr);
    EXPECT_NE(b, nullptr);
    // (There's only one codebook, not per-layer — this verifies the struct)
    EXPECT_EQ(res->d_c(), 448);

    destroy_tq_resources(*res, *backend);
}

TEST(TqResourcesLifecycle, PiPerLayer) {
    auto backend = make_null_backend();
    TqInitOptions opts;
    opts.d_c           = 64;  // small for speed
    opts.bits          = 4;
    opts.num_layers    = 4;
    opts.codebook_dir  = codebook_dir();
    opts.attention_device = backend.get();


    // Need d=64 codebook — create a minimal one in a temp directory
    namespace fs = std::filesystem;
    auto tmp_dir = fs::temp_directory_path() / "tq_test_codebooks";
    fs::create_directories(tmp_dir);
    auto tmp_cb = tmp_dir / "codebook_d64_b4.json";
    {
        std::ofstream f(tmp_cb);
        f << R"({
            "d": 64, "bits": 4, "n_clusters": 16,
            "centroids": [-0.3, -0.25, -0.2, -0.15, -0.1, -0.06, -0.03, -0.01,
                           0.01,  0.03,  0.06,  0.1,   0.15,  0.2,   0.25,  0.3],
            "boundaries": [-1.0, -0.275, -0.225, -0.175, -0.125, -0.08, -0.045, -0.02,
                            0.0, 0.02, 0.045, 0.08, 0.125, 0.175, 0.225, 0.275, 1.0]
        })";
    }
    opts.codebook_dir = tmp_dir.string();

    auto res = init_tq_resources(opts);

    // Each layer should have a distinct Π pointer
    std::set<const float*> pi_ptrs;
    for (int l = 0; l < 4; ++l) {
        const float* p = res->device_Pi(l);
        EXPECT_NE(p, nullptr);
        pi_ptrs.insert(p);
    }
    EXPECT_EQ(pi_ptrs.size(), 4u) << "Each layer should have distinct Π device pointer";

    destroy_tq_resources(*res, *backend);
    fs::remove_all(tmp_dir);
}

TEST(TqResourcesLifecycle, CentroidsReadBack) {
    // Verify codebook values are correctly uploaded (NullAttentionDevice uses malloc + memcpy)
    auto backend = make_null_backend();
    TqInitOptions opts;
    opts.d_c           = 512;
    opts.bits          = 4;
    opts.num_layers    = 1;
    opts.codebook_dir  = codebook_dir();
    opts.attention_device = backend.get();


    auto res = init_tq_resources(opts);

    // Since NullAttentionDevice uses malloc + memcpy, we can read back
    const float* d_c = res->device_centroids();
    ASSERT_NE(d_c, nullptr);
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(d_c[i], res->codebook.centroids[i])
            << "Centroid readback mismatch at index " << i;
    }

    const float* d_b = res->device_boundaries();
    ASSERT_NE(d_b, nullptr);
    for (int i = 0; i < 15; ++i) {
        EXPECT_FLOAT_EQ(d_b[i], res->codebook.interior_boundaries[i])
            << "Boundary readback mismatch at index " << i;
    }

    destroy_tq_resources(*res, *backend);
}

TEST(TqResourcesLifecycle, PiReadBack) {
    // Verify rotation matrix is correctly uploaded
    auto backend = make_null_backend();
    TqInitOptions opts;
    opts.d_c           = 512;
    opts.bits          = 4;
    opts.num_layers    = 1;
    opts.codebook_dir  = codebook_dir();
    opts.attention_device = backend.get();


    auto res = init_tq_resources(opts);

    const float* d_pi = res->device_Pi(0);
    ASSERT_NE(d_pi, nullptr);

    // Compare device copy against host copy
    const auto& host_pi = res->rotations[0].Pi;
    ASSERT_EQ(host_pi.size(), static_cast<size_t>(512 * 512));
    for (int i = 0; i < 512 * 512; ++i) {
        EXPECT_EQ(d_pi[i], host_pi[i]) << "Π readback mismatch at index " << i;
    }

    destroy_tq_resources(*res, *backend);
}

TEST(TqResourcesLifecycle, RejectsInvalidOptions) {
    auto backend = make_null_backend();
    TqInitOptions opts;
    opts.attention_device = backend.get();


    // d_c = 0
    opts.d_c = 0;
    opts.num_layers = 1;
    opts.codebook_dir = "/tmp";
    EXPECT_THROW(init_tq_resources(opts), std::runtime_error);

    // num_layers = 0
    opts.d_c = 512;
    opts.num_layers = 0;
    EXPECT_THROW(init_tq_resources(opts), std::runtime_error);

    // null backend
    opts.num_layers = 1;
    opts.attention_device = nullptr;
    EXPECT_THROW(init_tq_resources(opts), std::runtime_error);
}

#endif  // LAYERSTORM_SOURCE_DIR
