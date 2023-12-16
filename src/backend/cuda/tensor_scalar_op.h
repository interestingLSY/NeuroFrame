#pragma once

#include "src/tensor/tensor.h"
#include "src/basic/scalar.h"

namespace NeuroFrame::Backend::CUDA {

Tensor tensor_adds(const Tensor &input, const Scalar &Scalar);

Tensor tensor_subs(const Tensor &input, const Scalar &Scalar);

Tensor tensor_muls(const Tensor &input, const Scalar &Scalar);

Tensor tensor_divs(const Tensor &input, const Scalar &Scalar);

Tensor tensor_pows(const Tensor &input, const Scalar &Scalar);

}