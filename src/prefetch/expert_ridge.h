#pragma once
// ExpertRidge (P11) engine-side inference — the seam that consumes the
// published /srv/models/GLM-5.2-P11-ExpertRidge bundle (x18ship3:
// pipelined(2) leader).
//
// ADVISORY-ONLY (INV-ELB-ADVISORY): this module is a pure signal
// producer. Its outputs (pool ranking, eviction scores, manifest
// scores, pi-hat coverage curves) may inform WHEN weight bytes move
// (prefetch, eviction bias) but NEVER which experts a layer computes
// or any tensor value. It issues no fetches (INV-4.5) and is not
// wired into live fetch decisions — future P2/P9/eviction governors
// consume it behind their own gates (INV-ELB-CAP applies at issue
// time, not here).
//
// Semantics: full parity with the Python training engine
// (tools/elb_train/stagecprime_gpu.py) for the deployed subset —
// feature pass (base 12 + xprev/memo/memo2/ctx/xsame columns in
// feature_spec order), head evaluation (w_open / w_recur16 /
// w_manifest with the trained masked-column weights), pi-hat
// rank-conditional calibration (predict-then-update), and the online
// update rules (decayed A/b sufficient statistics with closed-form
// re-solve every RLS_REFRESH positions, pair-table lazy-decay EMAs,
// trail windows, calibration counts). The ctx encoder runs FROZEN
// (online Adam is training-time only — deployment contract).
//
// PIPELINED(2) causality (feature_spec.pipelined_protocol): the
// xsame feature row for layer j reads the ROUTED top-K of layer
// j-lag at the CURRENT position; rows j < lag are zero. step()
// consumes the full position (verify executes layers sequentially,
// so by the time any consumer acts on layer j's advice, layer
// j-lag's routing has landed); per-row causality is honored by
// construction inside the feature builder. Tables are updated
// strictly AFTER scoring (score-then-update).
//
// CPU-only, GPU-SDK-free (INV-GPU-1); scoring hot path is
// allocation-free after load() (INV-KEEPPRED discipline).

#include <cstdint>
#include <string>
#include <vector>

namespace layerstorm::prefetch {

// Per-position advisory outputs. Pointers view module-owned scratch,
// valid until the next step()/score() call. Layout row-major.
struct ExpertRidgeAdvice {
    const float* open_scores = nullptr;      // [J, E] w_open
    const float* evict_scores = nullptr;     // [J, E] w_recur16
    const float* manifest_scores = nullptr;  // [J, E] w_manifest
    const int32_t* pool_ids = nullptr;       // [J, pool_m - K] open
                                             // ranking, prev-exempt
    const float* cal_curve = nullptr;        // [J] pi-hat C_j(pool_m)
    int j = 0, e = 0, k = 0, pool_m = 0;
};

class ExpertRidge {
  public:
    // Load the published model dir (model.safetensors +
    // feature_spec.json authoritative). Returns false + err on
    // failure. Allocates ALL scratch here (hot path allocation-free).
    bool load(const std::string& model_dir, std::string* err);

    // Test/fixture initialisation: dimensions + frozen ctx params +
    // PCA table, everything else zeroed (the parity fixture replays
    // online training from scratch). `feat_names` = comma list in
    // feature_spec order.
    bool init_synthetic(int j, int e, int k, int memo_buckets,
                        int memo2_buckets, int xsame_lag, int pool_m,
                        const std::string& feat_names,
                        const std::vector<float>& pca,  // [V, d_tok]
                        int v, int d_tok, int ctx_d, int ctx_d_out,
                        std::string* err);

    // Reset per-sequence state (prev8/trails/EMAs/rings). Persistent
    // state (tables, ridge stats, calibration) carries across.
    void begin_sequence();

    // Frozen ctx-encoder parameters (fixture/test path; load() fills
    // them from the bundle). Shapes per stagecprime_attn.CtxEncoder.
    void set_ctx_params(std::vector<float> q, std::vector<float> k,
                        std::vector<float> v, std::vector<float> pos,
                        std::vector<float> proj_w,
                        std::vector<float> proj_b,
                        std::vector<float> exp_out,
                        std::vector<float> bias);

    // One committed position: compute the advisory outputs from
    // pre-update state (+ this position's sel/tops for the pipelined
    // xsame column and the pi-hat pre-update read), then apply every
    // update rule. sel [J,E] (stable_sigmoid(logits) + bias, the
    // engine routing sel space), tops [J,K] routed experts,
    // top_w [J,K] gate weights, token_bucket = embed_lut[token]
    // (identity lut: the token id), bigram_bucket = hashed
    // (prev_tok, tok), token_id for the ctx ring.
    // Score-only advisory read (NO state mutation): features from
    // state <= t-1 (+ this position's sel/tops for the pipelined
    // xsame column), heads, pool ranking, pi-hat curve. The scoring
    // hot path the rent budget measures.
    const ExpertRidgeAdvice& advise(const float* sel,
                                    const int32_t* tops,
                                    int32_t token_bucket,
                                    int32_t bigram_bucket,
                                    int32_t token_id);

    const ExpertRidgeAdvice& step(const float* sel,
                                  const int32_t* tops,
                                  const float* top_w,
                                  int32_t token_bucket,
                                  int32_t bigram_bucket,
                                  int32_t token_id);

    // Debug/parity accessors (module-owned scratch).
    const float* features_x() const { return x_.data(); }  // [J,E,F]
    int feature_count() const { return f_; }
    double cal_n() const { return cal_n_; }
    const std::vector<double>& w_open() const { return wo_; }
    const std::vector<double>& w_recur16() const { return wrec_; }
    const std::vector<double>& w_manifest() const { return wman_; }
    const std::vector<float>& trans_table() const { return trans_; }
    float trans_scale() const { return trans_scale_; }
    const std::vector<float>& xsame_table() const { return xsame_; }
    float xsame_scale() const { return xsame_scale_; }
    const std::vector<float>& cal_h() const { return calH_; }
    const std::vector<float>& cal_p() const { return calP_; }

  private:
    void build_features(const float* sel, const int32_t* tops,
                        int32_t bucket, int32_t bucket2,
                        int32_t token_id);
    void ctx_forward(int32_t token_id, float* out);   // [J, E]
    void score_heads();
    void cal_update(const float* sc_open, const int32_t* tops);
    void update_state(const float* sel, const int32_t* tops,
                      const float* top_w, int32_t bucket,
                      int32_t bucket2, int32_t token_id);
    void refresh_solve();

    // dims / config
    int j_ = 0, e_ = 0, k_ = 0, f_ = 0, fa_ = 0;  // fa_ = F+1
    int memo_b_ = 0, memo2_b_ = 0, xsame_lag_ = 0, pool_m_ = 32;
    int v_ = 0, d_tok_ = 0, ctx_d_ = 0, ctx_dout_ = 0;
    bool has_memo_ = false, has_memo2_ = false, has_ctx_ = false,
         has_xsame_ = false, has_xu_ = false, has_xd_ = false;
    std::vector<int> extra_cols_;   // column index per extra feature
    std::vector<std::string> feat_names_;
    std::vector<double> head_mask_;             // [F+1]

    // persistent tables (f32, lazy-decay scales)
    std::vector<float> trans_, xu_, xd_, xsame_;  // [J,E,E]
    float trans_scale_ = 1.f, xu_scale_ = 1.f, xd_scale_ = 1.f,
          xsame_scale_ = 1.f;
    std::vector<float> memo_, memo2_;             // dense [J,B,E]
    // ridge sufficient statistics + heads (f64)
    std::vector<double> A_, Ao_, Asub_, b_, bo_, brec_, bman_;
    std::vector<double> wr_, wo_, wrec_, wman_;
    int since_ = 0;
    long t_idx_ = 0;
    // recur ring (Xa f32 snapshots) + horizon
    std::vector<std::vector<float>> rring_;
    std::vector<long> rring_t_;
    // pi-hat
    std::vector<float> calH_, calP_;              // [J,E-K],[J,K]
    double cal_n_ = 0.0;
    // per-sequence state
    std::vector<float> prev8_mask_, t16_, t64_, ema2_, ema8_,
        g_prev_, g_prev2_, cut_prev_;
    std::vector<int32_t> prev8_;
    std::vector<long> last_seen_;
    double n2_ = 0.0, n8_ = 0.0;
    std::vector<std::vector<int32_t>> ring16_, ring64_;
    std::vector<int32_t> ctx_ring_;
    // frozen ctx encoder params + PCA
    std::vector<float> pca_, ctx_q_, ctx_k_, ctx_v_, ctx_pos_,
        ctx_projw_, ctx_projb_, ctx_expout_, ctx_bias_;
    // scratch (preallocated)
    std::vector<float> x_, ctx_now_, sc_open_f_, sc_rec_f_,
        sc_man_f_, tmp_e_, tmp_d_;
    std::vector<double> sc_open_, sc_rec_, sc_man_, xa_row_;
    std::vector<float> tp_, trank_;
    std::vector<int32_t> order_, pool_ids_;
    std::vector<float> cal_curve_;
    ExpertRidgeAdvice advice_;
};

}  // namespace layerstorm::prefetch
