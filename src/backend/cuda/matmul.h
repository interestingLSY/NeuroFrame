#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor matmul(const Tensor &input1, const Tensor &input2, bool transpose_a, bool transpose_b);

Tensor batched_matmul(const Tensor &input1, const Tensor &input2, bool transpose_a, bool transpose_b);

}