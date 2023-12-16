#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor transpose_forward_manual(const Tensor &input, OpContext &ctx, int axe1, int axe2);

Tensor transpose_backward_manual(const Tensor &output_grad, const OpContext &ctx);

Tensor transpose(const Tensor &input, int axe1, int axe2);

}