#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor cross_entropy_loss_forward_manual(const Tensor &input, const Tensor &ground_truth, OpContext &ctx);

Tensor cross_entropy_loss_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor cross_entropy_loss(const Tensor &input, const Tensor &ground_truth);

}