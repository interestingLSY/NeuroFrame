#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor sigmoid_forward_manual(const Tensor &input, OpContext &ctx);

Tensor sigmoid_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor sigmoid(const Tensor &input);

}