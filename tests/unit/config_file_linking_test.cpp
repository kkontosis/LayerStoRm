#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "config/config_preset.h"

using namespace layerstorm::config;
namespace fs = std::filesystem;

class ConfigFileLinkingTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "layerstorm_cfg_link_test";
        fs::create_directories(tmp_dir_);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir_);
    }

    void write_json(const std::string& filename, const nlohmann::json& j) {
        std::ofstream f(tmp_dir_ / filename);
        f << j.dump(2);
    }

    fs::path tmp_dir_;
};

TEST_F(ConfigFileLinkingTest, EmptyLinksNoOp) {
    nlohmann::json j = {{"model", {{"architecture", "deepseek_v3"}}}};
    resolve_config_file_links(j, tmp_dir_.string());
    EXPECT_FALSE(j.contains("_internal-prescope"));
}

TEST_F(ConfigFileLinkingTest, InlineSectionWinsOverFile) {
    nlohmann::json j = {
        {"_internal-prescope", {{"enabled", false}}}
    };
    write_json("prescope.json", {{"enabled", true}, {"top_k", 32}});
    resolve_config_file_links(j, tmp_dir_.string());
    EXPECT_TRUE(j.contains("_internal-prescope"));
    EXPECT_FALSE(j["_internal-prescope"]["enabled"].get<bool>());
    EXPECT_FALSE(j["_internal-prescope"].contains("top_k"));
}
