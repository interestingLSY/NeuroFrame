#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor tensor_negate(const Tensor &input);

}