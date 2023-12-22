#include "sgd.h"

#include "src/basic/inference_mode.h"
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
	InferenceModeGuard guard;
	guard.__enter__();
	for (Tensor &tensor : focused_nodes) {
		const std::shared_ptr<CGraph::CGraphNode> &node = tensor.cgraph_node;
		if (!node->grad) {
			LOG_FATAL("Gradient of one tensor is not available during gradient descent");
		}
		Tensor new_weight = tensor_sub(
			tensor,
			tensor_muls(
				node->grad.value(),
				Scalar(learning_rate, tensor.dtype)
			)
		);
		tensor.mem_frag.copy_from(new_weight.mem_frag);
	}
	guard.__exit__();
}

}
