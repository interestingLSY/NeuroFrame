#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

Tensor transpose(const Tensor &input, int axe1, int axe2);

}