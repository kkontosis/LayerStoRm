#pragma once

//==============================================================================
// Decode CUDA Graph Runners — type aliases
//
// Brings sm120::graph::{DecodeGraphRunner, TqDecodeGraphRunner} into the
// layerstorm::compute namespace.
//==============================================================================

#include "sm120/graph/decode_graph.h"
#include "sm120/graph/tq_decode_graph.h"

namespace layerstorm::compute {

using DecodeGraphRunner = sm120::graph::DecodeGraphRunner;
using DecodeGraphConfig = sm120::graph::DecodeGraphConfig;

using TqDecodeGraphRunner = sm120::graph::TqDecodeGraphRunner;
using TqDecodeGraphConfig = sm120::graph::TqDecodeGraphConfig;

}  // namespace layerstorm::compute
