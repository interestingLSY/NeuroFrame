#pragma once

#include "src/tensor/tensor.h"
#include "src/basic/scalar.h"

namespace NeuroFrame::Backend::CPU {

Tensor tensor_adds(const Tensor &input, const Scalar &Scalar);

}