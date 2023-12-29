#pragma once

#include "op.h"

#include <vector>

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor pool_forward_manual(const Tensor &input, int64_t pool_size, int64_t stride, int64_t padding, OpContext &ctx);

Tensor pool_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor pool(const Tensor &input, int64_t pool_size, int64_t stride, int64_t padding);

}