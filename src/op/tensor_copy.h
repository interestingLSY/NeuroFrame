#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor tensor_copy_forward_manual(const Tensor &input, OpContext &ctx);

Tensor tensor_copy(const Tensor &input);

}