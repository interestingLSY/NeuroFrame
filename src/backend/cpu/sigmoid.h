#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

Tensor sigmoid_forward(const Tensor &input);

Tensor sigmoid_backward(const Tensor &output_grad, const Tensor &output);

}