#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

std::pair<Tensor, Tensor> pool_forward(const Tensor &input, int pool_size);

Tensor pool_backward(const Tensor &output_grad, const Tensor &max_mask, int pool_size);

}