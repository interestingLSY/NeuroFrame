#pragma once

#include <memory>

#include "src/tensor/tensor.h"
#include "src/op/op_context.h"
#include "src/op/op.h"

#include "cgraph_node.h"

namespace NeuroFrame::CGraph {

void on_new_calculation(
	std::vector<Tensor> &inputs,
	std::vector<Tensor> &outputs,
	const OpContext &ctx,
	op_backward_func_t backward_func
);

std::vector<Tensor> perform_backward(Tensor &src, Tensor &src_grad, bool log_down_all_grads);

Tensor get_computed_grad(const Tensor &src);

// std::vector<Tensor> get_topology_sort(Tensor &src);

void clear_graph();

}