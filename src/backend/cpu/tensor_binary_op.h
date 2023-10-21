#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

Tensor tensor_add(const Tensor &input1, const Tensor &input2);

Tensor tensor_sub(const Tensor &input1, const Tensor &input2);

Tensor tensor_mul(const Tensor &input1, const Tensor &input2);

Tensor tensor_div(const Tensor &input1, const Tensor &input2);

}