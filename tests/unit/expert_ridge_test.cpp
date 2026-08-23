// ExpertRidge (P11) engine-side inference — parity vs the Python
// training engine. Golden fixture tests/assets/expert_ridge_parity.
// safetensors is produced by the E2E pipeline step stagecprime_cppfix
// (tools/elb_train/export_cpp_fixture.py): a deterministic small-dims
// replay of the full x18ship3 semantics. The C++ module replays the
// same inputs and must match every checkpoint (fp32-class tolerance)
// and the final persistent state.

#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "model/weight_loader/safetensors_reader.h"
#include "prefetch/expert_ridge.h"

namespace {

using layerstorm::model::SafetensorsReader;
using layerstorm::model::TensorEntry;
using layerstorm::prefetch::ExpertRidge;

class Fixture {
  public:
    Fixture() {
        const std::string path =
            std::string(LAYERSTORM_SOURCE_DIR) +
            "/tests/assets/expert_ridge_parity.safetensors";
        rd_ = SafetensorsReader::open(path, /*use_mmap=*/true);
    }
    const TensorEntry* find(const std::string& n) const {
        for (const auto& e : rd_.entries())
            if (e.name == n) return &e;
        return nullptr;
    }
    template <typename T>
    std::vector<T> get(const std::string& n) const {
        const auto* e = find(n);
        EXPECT_NE(e, nullptr) << n;
        if (!e) return {};
        auto sp = rd_.tensor_data(*e);
        std::vector<T> out(sp.size() / sizeof(T));
        std::memcpy(out.data(), sp.data(), sp.size());
        return out;
    }
    long geti(const std::string& n) const {
        return get<int64_t>(n).at(0);
    }
    std::string get_csv(const std::string& n) const {
        auto raw = get<uint8_t>(n + "_csv");
        return std::string(raw.begin(), raw.end());
    }

  private:
    SafetensorsReader rd_;
};

class ExpertRidgeParity : public ::testing::Test {
  protected:
    void SetUp() override {
        fx_ = std::make_unique<Fixture>();
        J = static_cast<int>(fx_->geti("J"));
        E = static_cast<int>(fx_->geti("E"));
        K = static_cast<int>(fx_->geti("K"));
        n_pos = static_cast<int>(fx_->geti("n_pos"));
        pool_m = static_cast<int>(fx_->geti("pool_m"));
        auto pca = fx_->get<float>("pca");
        std::string err;
        ASSERT_TRUE(er_.init_synthetic(
            J, E, K, static_cast<int>(fx_->geti("memo_buckets")),
            static_cast<int>(fx_->geti("memo2_buckets")),
            static_cast<int>(fx_->geti("xsame_lag")), pool_m,
            fx_->get_csv("feat_names"), pca,
            static_cast<int>(fx_->geti("V")),
            static_cast<int>(fx_->geti("d_tok")), 48, 32, &err))
            << err;
        // frozen ctx params
        load_ctx("ctx_q_weight", "ctx_k_weight", "ctx_v_weight",
                 "ctx_pos_weight", "ctx_proj_weight",
                 "ctx_proj_bias", "ctx_exp_out", "ctx_bias");
        tops_ = fx_->get<int64_t>("in_tops");
        sel_ = fx_->get<float>("in_sel");
        topw_ = fx_->get<float>("in_top_w");
        bucket_ = fx_->get<int64_t>("in_bucket");
        bucket2_ = fx_->get<int64_t>("in_bucket2");
        tok_ = fx_->get<int64_t>("in_tok");
    }

    void load_ctx(const char* q, const char* k, const char* v,
                  const char* pos, const char* pw, const char* pb,
                  const char* eo, const char* b) {
        er_.set_ctx_params(fx_->get<float>(q), fx_->get<float>(k),
                           fx_->get<float>(v), fx_->get<float>(pos),
                           fx_->get<float>(pw), fx_->get<float>(pb),
                           fx_->get<float>(eo), fx_->get<float>(b));
    }

    // replay one position t; returns the advice
    const layerstorm::prefetch::ExpertRidgeAdvice& step(int t) {
        std::vector<int32_t> tops32(static_cast<size_t>(J) * K);
        for (size_t i = 0; i < tops32.size(); ++i)
            tops32[i] = static_cast<int32_t>(
                tops_[static_cast<size_t>(t) * J * K + i]);
        tops_scratch_ = tops32;
        return er_.step(
            &sel_[static_cast<size_t>(t) * J * E],
            tops_scratch_.data(),
            &topw_[static_cast<size_t>(t) * J * K],
            static_cast<int32_t>(bucket_[t]),
            static_cast<int32_t>(bucket2_[t]),
            static_cast<int32_t>(tok_[t]));
    }

    std::unique_ptr<Fixture> fx_;
    ExpertRidge er_;
    int J = 0, E = 0, K = 0, n_pos = 0, pool_m = 0;
    std::vector<int64_t> tops_, bucket_, bucket2_, tok_;
    std::vector<float> sel_, topw_;
    std::vector<int32_t> tops_scratch_;
};

TEST_F(ExpertRidgeParity, FullReplayMatchesPythonEngine) {
    auto checks = fx_->get<int64_t>("checks");
    double max_x = 0, max_sc = 0, max_cal = 0;
    long pool_mismatch = 0;
    double total_us = 0;
    for (int t = 0; t < n_pos; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& adv = step(t);
        total_us += std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        const bool is_check =
            std::find(checks.begin(), checks.end(), t) != checks.end();
        if (!is_check) continue;
        const std::string cp = "t" + std::to_string(t);
        auto gx = fx_->get<float>(cp + "_X");
        const float* cx = er_.features_x();
        for (size_t i = 0; i < gx.size(); ++i)
            max_x = std::max(
                max_x, static_cast<double>(std::fabs(gx[i] - cx[i])));
        auto gso = fx_->get<double>(cp + "_sc_open");
        auto gsr = fx_->get<double>(cp + "_sc_rec");
        auto gsm = fx_->get<double>(cp + "_sc_man");
        for (int i = 0; i < J * E; ++i) {
            max_sc = std::max(
                max_sc, std::fabs(gso[i] - adv.open_scores[i]));
            max_sc = std::max(
                max_sc, std::fabs(gsr[i] - adv.evict_scores[i]));
            max_sc = std::max(
                max_sc, std::fabs(gsm[i] - adv.manifest_scores[i]));
        }
        auto gpool = fx_->get<int64_t>(cp + "_pool_open");
        for (size_t i = 0; i < gpool.size(); ++i)
            if (gpool[i] != adv.pool_ids[i]) ++pool_mismatch;
        auto gcal = fx_->get<double>(cp + "_cal_curve");
        for (int jj = 0; jj < J; ++jj)
            max_cal = std::max(
                max_cal, std::fabs(gcal[jj] - adv.cal_curve[jj]));
    }
    EXPECT_LT(max_x, 2e-3) << "feature matrix drift";
    EXPECT_LT(max_sc, 5e-3) << "head score drift";
    EXPECT_EQ(pool_mismatch, 0) << "pool ranking drift";
    EXPECT_LT(max_cal, 1e-4) << "pi-hat curve drift";
    // final persistent state
    auto fwo = fx_->get<double>("final_wo");
    auto fwr = fx_->get<double>("final_wrec16");
    auto fwm = fx_->get<double>("final_wman");
    double max_w = 0;
    for (size_t i = 0; i < fwo.size(); ++i) {
        max_w = std::max(max_w, std::fabs(fwo[i] - er_.w_open()[i]));
        max_w = std::max(max_w,
                         std::fabs(fwr[i] - er_.w_recur16()[i]));
        max_w = std::max(max_w,
                         std::fabs(fwm[i] - er_.w_manifest()[i]));
    }
    EXPECT_LT(max_w, 1e-4) << "solved head weight drift";
    auto ftr = fx_->get<float>("final_trans");
    auto fxs = fx_->get<float>("final_xsame");
    double max_t = 0;
    for (size_t i = 0; i < ftr.size(); ++i) {
        max_t = std::max(
            max_t, static_cast<double>(std::fabs(
                       ftr[i] - er_.trans_table()[i] *
                                    er_.trans_scale())));
        max_t = std::max(
            max_t, static_cast<double>(std::fabs(
                       fxs[i] - er_.xsame_table()[i] *
                                    er_.xsame_scale())));
    }
    EXPECT_LT(max_t, 2e-3) << "pair table drift";
    auto fch = fx_->get<float>("final_calH");
    auto fcp = fx_->get<float>("final_calP");
    double max_c = 0;
    for (size_t i = 0; i < fch.size(); ++i)
        max_c = std::max(max_c, static_cast<double>(std::fabs(
                                    fch[i] - er_.cal_h()[i])));
    for (size_t i = 0; i < fcp.size(); ++i)
        max_c = std::max(max_c, static_cast<double>(std::fabs(
                                    fcp[i] - er_.cal_p()[i])));
    EXPECT_LT(max_c, 1e-5) << "calibration table drift";
    EXPECT_NEAR(er_.cal_n(), fx_->get<double>("final_cal_n")[0],
                1e-12);
    // informational: per-position / per-layer cost at fixture dims
    ::testing::Test::RecordProperty(
        "us_per_position_fixture_dims",
        static_cast<int>(total_us / n_pos));
}

TEST_F(ExpertRidgeParity, SequenceResetClearsPerSeqState) {
    for (int t = 0; t < 8; ++t) step(t);
    er_.begin_sequence();
    const auto& adv = step(0);   // t_idx back to 0 -> zero advice
    for (int i = 0; i < J * E; ++i)
        EXPECT_EQ(adv.open_scores[i], 0.f);
}

}  // namespace
