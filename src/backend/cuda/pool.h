#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor pool_forward(const Tensor &input, int pool_size, int stride, int padding);

Tensor pool_backward(const Tensor &output_grad, const Tensor &input, const Tensor &output, int pool_size, int stride, int padding);

}