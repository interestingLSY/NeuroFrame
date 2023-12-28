#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

Tensor tensor_reduction_sum(const Tensor &input, int axis);

Tensor tensor_reduction_max(const Tensor &input, int axis);

Tensor tensor_reduction_min(const Tensor &input, int axis);

}