#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor broadcast_to_forward_manual(const Tensor &input, OpContext &ctx, const std::vector<int64_t> &shape);

Tensor broadcast_to_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor broadcast_to(const Tensor &input, const std::vector<int64_t> &shape);

}