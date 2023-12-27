#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

Tensor batch_norm(const Tensor &input, const Tensor &gamma, const Tensor &beta, const Tensor &mean, const Tensor &variance, double momentum, double epsilon);

}