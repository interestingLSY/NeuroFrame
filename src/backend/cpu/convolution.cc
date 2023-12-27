#include "convolution.h"

#include "src/basic/log.h"
#include "src/tensor/tensor.h"

#include "utils.h"

namespace NeuroFrame::Backend::CPU {

Tensor convolution_forward(const Tensor &input_img, const Tensor &kernel, const int64_t stride, const int64_t dilation) {
	LOG_FATAL("convolution_forward is not implemented for CPU");
}

std::tuple<Tensor, Tensor> convolution_backward(const Tensor &output_grad, const Tensor &input_img, const Tensor &kernel, const int64_t stride, const int64_t dilation) {
	LOG_FATAL("convolution_backward is not implemented for CPU");
}


}