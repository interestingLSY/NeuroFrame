#include "sgd.h"

#include "src/op/tensor_binary_op.h"
#include "src/op/tensor_scalar_op.h"

namespace NeuroFrame {

SGDOptimState::SGDOptimState() {}

SGDOptimState::~SGDOptimState() {}

SGDOptimizer::SGDOptimizer() {}

SGDOptimizer::~SGDOptimizer() {
	for (const Tensor &tensor : focused_nodes) {
		remove_focus(tensor);
	}
}

void SGDOptimizer::add_focus(const Tensor &tensor) {
	std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
	add_to_focus_list(tensor);
	node->reset_optim_state();
	node->optim_state = std::make_shared<SGDOptimState>();
}

void SGDOptimizer::remove_focus(const Tensor &tensor) {
	std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
	remove_from_focus_list(tensor);
	node->reset_optim_state();
}

void SGDOptimizer::step(double learning_rate) {
	for (Tensor &tensor : focused_nodes) {
		std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
		std::shared_ptr<SGDOptimState> state = std::static_pointer_cast<SGDOptimState>(node->optim_state);
		if (!tensor.cgraph_node->grad) {
			LOG_FATAL("Gradient of one tensor is not available during gradient descent");
		}
		OpContext temp_ctx;
		Tensor new_weight = tensor_sub_forward_manual(
			tensor,
			tensor_muls_forward_manual(
				tensor.cgraph_node->grad.value(),
				Scalar(learning_rate, tensor.dtype),
				temp_ctx
			),
			temp_ctx
		);
		tensor.mem_frag.copy_from(new_weight.mem_frag);
	}
}

}
