#pragma once

#include "src/tensor/tensor.h"
#include "src/basic/scalar.h"

namespace NeuroFrame::Backend::CUDA {

Tensor tensor_adds(const Tensor &input, const Scalar &Scalar);

}