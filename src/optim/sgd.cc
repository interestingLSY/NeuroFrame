#include "sgd.h"

#include "src/basic/inference_mode.h"
#include "src/op/tensor_binary_op.h"
#include "src/op/tensor_scalar_op.h"
#include "src/op/misc.h"

namespace NeuroFrame {

SGDOptimState::SGDOptimState(const Tensor &weight, bool have_momentum):
	momentum(have_momentum ? Tensor::zeros_like(weight) : Tensor({}, weight.dtype, weight.device)) {}

SGDOptimState::~SGDOptimState() {}

SGDOptimizer::SGDOptimizer(double momentum, double weight_decay):
	momentum(momentum),
	weight_decay(weight_decay) {}

SGDOptimizer::~SGDOptimizer() {
	for (const Tensor &tensor : focused_nodes) {
		remove_focus(tensor);
	}
}

void SGDOptimizer::add_focus(const Tensor &tensor) {
	std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
	add_to_focus_list(tensor);
	node->reset_optim_state();
	node->optim_state = std::make_shared<SGDOptimState>(tensor, momentum != 0.0);
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
		std::shared_ptr<SGDOptimState> optim_state = std::static_pointer_cast<SGDOptimState>(node->optim_state);
		if (!node->grad) {
			LOG_FATAL("Gradient of one tensor is not available during gradient descent");
		}
		sgd_grad_update(
			tensor,
			node->grad.value(),
			optim_state->momentum,
			learning_rate,
			momentum,
			weight_decay
		);
	}
	guard.__exit__();
}

}
