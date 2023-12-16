#pragma once

#include "op.h"

#include "src/tensor/tensor.h"
#include "src/basic/scalar.h"

namespace NeuroFrame {

Tensor tensor_adds_forward_manual(const Tensor &a, const Scalar &b, OpContext &ctx);

Tensor tensor_adds(const Tensor &a, const Scalar &b);

Tensor tensor_subs_forward_manual(const Tensor &a, const Scalar &b, OpContext &ctx);

Tensor tensor_subs(const Tensor &a, const Scalar &b);

Tensor tensor_muls_forward_manual(const Tensor &a, const Scalar &b, OpContext &ctx);

Tensor tensor_muls(const Tensor &a, const Scalar &b);

Tensor tensor_divs_forward_manual(const Tensor &a, const Scalar &b, OpContext &ctx);

Tensor tensor_divs(const Tensor &a, const Scalar &b);

Tensor tensor_pows_forward_manual(const Tensor &a, const Scalar &b, OpContext &ctx);

Tensor tensor_pows(const Tensor &a, const Scalar &b);

}