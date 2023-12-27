#pragma once

#include "op.h"

#include <vector>

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor batched_convolution(const Tensor &input, const Tensor &kernel, int stride, int dilation);

}