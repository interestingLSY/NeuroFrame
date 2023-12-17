#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor tensor_reduction_sum(const Tensor &input, int axis);

}