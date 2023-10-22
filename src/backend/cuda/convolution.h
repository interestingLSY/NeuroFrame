#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor batched_im2col(const Tensor &input_img, const int64_t kh, const int64_t kw);

Tensor batched_col2im(const Tensor &input, const int64_t c_in, const int64_t h, const int64_t w, const int64_t kh, const int64_t kw);

}