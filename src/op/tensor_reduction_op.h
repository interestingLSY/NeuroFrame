#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor tensor_reduction_sum_forward_manual(const Tensor &a, OpContext &ctx, int axis);

Tensor tensor_reduction_sum(const Tensor &a, int axis);

Tensor tensor_reduction_max(const Tensor &a, int axis);

Tensor tensor_reduction_min(const Tensor &a, int axis);

}