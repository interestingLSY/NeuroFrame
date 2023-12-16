#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor reshape_forward_manual(const Tensor &input, OpContext &ctx, const std::vector<int64_t> &shape);

Tensor reshape_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor reshape(const Tensor &input, const std::vector<int64_t> &shape);

}