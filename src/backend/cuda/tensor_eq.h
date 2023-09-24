#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

bool tensor_eq(const Tensor &input1, const Tensor &input2);

}