#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor transpose_forward_manual(const Tensor &input, OpContext &ctx);

Tensor transpose_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor transpose(const Tensor &input);

}