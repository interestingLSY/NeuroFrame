#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor tensor_negate(const Tensor &input);

Tensor tensor_inv(const Tensor &input);

Tensor tensor_exp(const Tensor &input);

Tensor tensor_log(const Tensor &input);

}