#pragma once

#include <tuple>

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor convolution_forward(const Tensor &input_img, const Tensor &kernel, const int64_t stride, const int64_t dilation);

std::tuple<Tensor, Tensor> convolution_backward(const Tensor &output_grad, const Tensor &input_img, const Tensor &kernel, const int64_t stride, const int64_t dilation);

}