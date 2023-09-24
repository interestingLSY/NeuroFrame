#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

// Call relu_forward manually. This only calculate the forward pass without
// altering the computation graph.
Tensor relu_forward_manual(const Tensor &input, OpContext &ctx);

// Call relu_backward manually. This only calculate the backward pass without
// altering the computation graph.
Tensor relu_backward_manual(const Tensor &output_grad, const OpContext &ctx);

// The most important function. User calls this in most scenarios.
Tensor relu(const Tensor &input);

}