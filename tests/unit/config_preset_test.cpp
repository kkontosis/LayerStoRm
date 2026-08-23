#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "config/config_parser.h"
#include "config/config_preset.h"

using namespace layerstorm::config;

// ── Helpers ──────────────────────────────────────────────────────────────────

static nlohmann::json minimal_json() {
    return {
        {"model", {
            {"architecture",           "deepseek_v3"},
            {"weights_path",           "/data/models/test/"},
            {"weights_format",         "safetensors"},
            {"num_hidden_layers",      61},
            {"hidden_size",            7168},
            {"num_attention_heads",    128},
            {"num_key_value_heads",    128},
            {"intermediate_size",      18432},
            {"n_routed_experts",       256},
            {"num_experts_per_tok",    8},
            {"vocab_size",             129280},
            {"max_position_embeddings", 163840},
        }},
        {"quantization", {
            {"weights",           "nvfp4"},
            {"attention_compute", "fp8_e4m3"},
            {"kv_cache",          "fp8_e4m3"},
            {"gating_compute",    "fp32"},
        }},
        {"hardware", {
            {"gpus", {{
                {"id",       0},
                {"type",     "rtx5090"},
                {"vram_gb",  32},
            }}},
            {"system_ram_gb", 256},
        }},
    };
}

#ifdef LAYERSTORM_SOURCE_DIR
static std::string presets_dir() {
    return std::string(LAYERSTORM_SOURCE_DIR) + "/config/presets";
}
#endif

// ── Tests: preset field in generated parser ─────────────────────────────────

TEST(ConfigPreset, PresetFieldParsed) {
    auto j = minimal_json();
    j["preset"] = "balanced";
    auto cfg = parse_config(j);
    ASSERT_TRUE(cfg.preset.has_value());
    EXPECT_EQ(*cfg.preset, "balanced");

    // Round-trip
    auto out = config_to_json(cfg);
    EXPECT_EQ(out["preset"], "balanced");
}

TEST(ConfigPreset, NoPresetDefault) {
    auto j = minimal_json();
    auto cfg = parse_config(j);
    EXPECT_FALSE(cfg.preset.has_value());

    // Serializes as null
    auto out = config_to_json(cfg);
    EXPECT_TRUE(out["preset"].is_null());
}

// ── Tests: JSON merge semantics ─────────────────────────────────────────────

TEST(ConfigPreset, PresetValuesAppliedWhenUserOmits) {
    // Simulate what load_config does: preset.merge_patch(user)
    nlohmann::json preset = {
        {"orchestrator", {{"max_batch_size", 256}}}
    };
    auto user = minimal_json();
    // User does NOT set orchestrator
    preset.merge_patch(user);
    auto cfg = parse_config(preset);
    EXPECT_EQ(cfg.orchestrator.max_batch_size, 256);  // from preset
}

TEST(ConfigPreset, UserOverridesPreset) {
    nlohmann::json preset = {
        {"orchestrator", {{"max_batch_size", 256}}}
    };
    auto user = minimal_json();
    user["orchestrator"] = {{"max_batch_size", 32}};
    preset.merge_patch(user);
    auto cfg = parse_config(preset);
    EXPECT_EQ(cfg.orchestrator.max_batch_size, 32);  // user wins
}

// ── Tests: validation ───────────────────────────────────────────────────────

TEST(ConfigPreset, ForbiddenSectionRejected_Model) {
    nlohmann::json bad = {{"model", {{"hidden_size", 4096}}}};
    EXPECT_THROW(validate_preset_json(bad, "bad"), std::runtime_error);
}

TEST(ConfigPreset, ForbiddenSectionRejected_Quantization) {
    nlohmann::json bad = {{"quantization", {{"weights", "fp8_e4m3"}}}};
    EXPECT_THROW(validate_preset_json(bad, "bad"), std::runtime_error);
}

TEST(ConfigPreset, ForbiddenSectionRejected_Hardware) {
    nlohmann::json bad = {{"hardware", {{"system_ram_gb", 64}}}};
    EXPECT_THROW(validate_preset_json(bad, "bad"), std::runtime_error);
}

TEST(ConfigPreset, PresetRecursionRejected) {
    nlohmann::json recursive = {
        {"preset", "other"},
        {"orchestrator", {{"max_batch_size", 128}}}
    };
    EXPECT_THROW(validate_preset_json(recursive, "recursive"),
                 std::runtime_error);
}

TEST(ConfigPreset, ValidPresetAccepted) {
    nlohmann::json good = {
        {"orchestrator", {{"max_batch_size", 128}}}
    };
    EXPECT_NO_THROW(validate_preset_json(good, "good"));
}

// ── Tests: load_preset ──────────────────────────────────────────────────────

TEST(ConfigPreset, UnknownPresetRejected) {
    EXPECT_THROW(
        load_preset("nonexistent_preset_xyz", {"/tmp/no_such_dir_abc"}),
        std::runtime_error);
}

#ifdef LAYERSTORM_SOURCE_DIR

TEST(ConfigPreset, LoadPresetFromDisk) {
    auto preset = load_preset("balanced", {presets_dir()});
    EXPECT_TRUE(preset.contains("orchestrator"));
    EXPECT_FALSE(preset.contains("model"));
}

// ── Tests: load_config integration ──────────────────────────────────────────

TEST(ConfigPreset, LoadConfigWithPreset) {
    auto user = minimal_json();
    user["preset"] = "balanced";
    auto cfg = load_config(user, {presets_dir()});

    // The balanced preset sets max_batch_size=64
    EXPECT_EQ(cfg.orchestrator.max_batch_size, 64);
    // preset field is erased before parse, so it's nullopt
    EXPECT_FALSE(cfg.preset.has_value());
}

TEST(ConfigPreset, LoadConfigUserOverridesPreset) {
    auto user = minimal_json();
    user["preset"] = "balanced";
    user["orchestrator"] = {{"max_batch_size", 16}};
    auto cfg = load_config(user, {presets_dir()});

    // User override wins over preset's 64
    EXPECT_EQ(cfg.orchestrator.max_batch_size, 16);
}

TEST(ConfigPreset, LoadConfigNoPreset) {
    auto user = minimal_json();
    auto cfg = load_config(user, {presets_dir()});

    // Without preset, defaults come from schema
    EXPECT_EQ(cfg.orchestrator.max_batch_size, 64);  // schema default
    EXPECT_FALSE(cfg.preset.has_value());
}

TEST(ConfigPreset, LoadConfigFromFile) {
    namespace fs = std::filesystem;

    // Write a temp config file that references a preset
    auto tmp_dir = fs::temp_directory_path() / "layerstorm_preset_test";
    fs::create_directories(tmp_dir / "presets");

    // Copy the balanced preset to the temp presets dir
    {
        auto src = fs::path(presets_dir()) / "balanced.json";
        auto dst = tmp_dir / "presets" / "balanced.json";
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
    }

    // Write the user config
    auto config_path = tmp_dir / "test_config.json";
    {
        auto user = minimal_json();
        user["preset"] = "balanced";
        user["orchestrator"] = {{"max_batch_size", 8}};
        std::ofstream f(config_path);
        f << user.dump(2);
    }

    auto cfg = load_config(config_path.string());
    EXPECT_EQ(cfg.orchestrator.max_batch_size, 8);  // user override
    EXPECT_FALSE(cfg.preset.has_value());

    // Cleanup
    fs::remove_all(tmp_dir);
}

// GLM-25d: the GLM-5.2 GGUF preset parses + carries the DSA/MLA/MoE dims.
TEST(ConfigPreset, Glm52GgufPresetParses) {
    namespace fs = std::filesystem;
    auto path = fs::path(LAYERSTORM_SOURCE_DIR) /
        "test-data/config/glm_5_2_gguf.json";
    ASSERT_TRUE(fs::exists(path)) << path;

    auto cfg = parse_config(path.string());
    const auto& m = cfg.model;
    EXPECT_EQ(m.architecture, Architecture::glm_moe_dsa);
    EXPECT_EQ(m.num_hidden_layers, 78);
    EXPECT_EQ(m.hidden_size, 6144);
    EXPECT_EQ(m.num_attention_heads, 64);
    // MLA-256 split
    EXPECT_EQ(m.qk_nope_head_dim, 192);
    EXPECT_EQ(m.qk_rope_head_dim, 64);
    EXPECT_EQ(m.v_head_dim, 256);
    EXPECT_EQ(m.kv_lora_rank, 512);
    EXPECT_EQ(m.q_lora_rank, 2048);
    // MoE (simple top-8, sigmoid)
    EXPECT_EQ(m.n_routed_experts, 256);
    EXPECT_EQ(m.num_experts_per_tok, 8);
    EXPECT_EQ(m.n_group, 1);
    EXPECT_EQ(m.first_k_dense_replace, 3);
    // DSA indexer — index_n_heads is 32 for GLM-5.2 (verified from GGUF)
    EXPECT_EQ(m.index_topk, 2048);
    EXPECT_EQ(m.index_n_heads, 32);
    EXPECT_EQ(m.index_head_dim, 128);
    EXPECT_TRUE(m.indexer_rope_interleave);
}

#endif  // LAYERSTORM_SOURCE_DIR

// ── memory.arena_attach.on_conflict × persist rejected at PARSE time ─────────
// (P-24b: finalize_config throws on the contradiction, so every production
// load_config path fails before any holder attach is attempted.)

TEST(ConfigPreset, ArenaOnConflictKillUnderPersistFailsAtParse) {
    auto j = minimal_json();
    j["memory"]["arena_attach"]["persist"] = true;
    j["memory"]["arena_attach"]["on_conflict"] = "kill";
    EXPECT_THROW(load_config(j, {}), std::runtime_error);
}

TEST(ConfigPreset, ArenaOnConflictNewUnderPersistFailsAtParse) {
    auto j = minimal_json();
    j["memory"]["arena_attach"]["persist"] = true;
    j["memory"]["arena_attach"]["on_conflict"] = "new";
    EXPECT_THROW(load_config(j, {}), std::runtime_error);
}

TEST(ConfigPreset, ArenaOnConflictCompatibleCombinationsLoad) {
    {  // persist=true + 'fail' — the only persist-compatible explicit mode
        auto j = minimal_json();
        j["memory"]["arena_attach"]["persist"] = true;
        j["memory"]["arena_attach"]["on_conflict"] = "fail";
        EXPECT_NO_THROW(load_config(j, {}));
    }
    {  // persist=true + absent (derives to 'fail')
        auto j = minimal_json();
        j["memory"]["arena_attach"]["persist"] = true;
        EXPECT_NO_THROW(load_config(j, {}));
    }
    for (const char* mode : {"new", "fail", "kill"}) {  // persist=false + all
        auto j = minimal_json();
        j["memory"]["arena_attach"]["persist"] = false;
        j["memory"]["arena_attach"]["on_conflict"] = mode;
        EXPECT_NO_THROW(load_config(j, {}));
    }
}
