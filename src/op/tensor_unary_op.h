#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor tensor_negate_forward_manual(const Tensor &a, OpContext &ctx);

Tensor tensor_negate(const Tensor &a);

Tensor tensor_inv_forward_manual(const Tensor &a, OpContext &ctx);

Tensor tensor_inv(const Tensor &a);

Tensor tensor_exp_forward_manual(const Tensor &a, OpContext &ctx);

Tensor tensor_exp(const Tensor &a);

Tensor tensor_log_forward_manual(const Tensor &a, OpContext &ctx);

Tensor tensor_log(const Tensor &a);

}