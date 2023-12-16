#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor broadcast_to(const Tensor &input, const std::vector<int64_t> &target_shape);

}