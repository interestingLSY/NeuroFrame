#pragma once

#include "op.h"

#include <vector>

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor batched_convolution_forward_manual(const Tensor &input_img, const Tensor &kernel, OpContext &ctx);

std::vector<Tensor> batched_convolution_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor batched_convolution(const Tensor &input, const Tensor &kernel);

}