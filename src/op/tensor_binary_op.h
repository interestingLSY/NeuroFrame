#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor tensor_add_forward_manual(const Tensor &a, const Tensor &b, OpContext &ctx);

Tensor tensor_add(const Tensor &a, const Tensor &b);

Tensor tensor_sub_forward_manual(const Tensor &a, const Tensor &b, OpContext &ctx);

Tensor tensor_sub(const Tensor &a, const Tensor &b);

}