#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

Tensor tensor_negate(const Tensor &input);

}