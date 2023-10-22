#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor batched_softmax_cross_entropy_loss(const Tensor& answer, const Tensor& ground_truth);

}
