#pragma once

#include "op.h"

#include <vector>

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor matmul_forward_manual(const Tensor &a, const Tensor &b, bool transpose_a, bool transpose_b, OpContext &ctx);

std::vector<Tensor> matmul_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor matmul(const Tensor &a, const Tensor &b, bool transpose_a, bool transpose_b);

}