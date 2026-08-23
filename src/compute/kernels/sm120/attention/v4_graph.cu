// Single-TU driver for the DeepSeek-V4 FP8 decode graph runner (V4-5e).
//
// The deps graph runner .cu defines the CsaFp8DecodeGraphRunner method
// bodies inline (single-TU include pattern, same as snapmla_prep.cu /
// lightning_indexer.cu); the decode kernel instantiations and mla_combine
// it captures are compiled separately (snapmla_kernels OBJECT lib).
// Consumed by src/compute/csa_hca_sm120_attention_device.cpp via
// <sm120/graph/csa_fp8_decode_graph.h>.

#include "sm120/graph/csa_fp8_decode_graph.cu"
