#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor tensor_negate_forward_manual(const Tensor &a, OpContext &ctx);

Tensor tensor_negate(const Tensor &a);

}