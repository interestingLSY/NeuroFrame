#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

bool tensor_eq(const Tensor &input1, const Tensor &input2);

}