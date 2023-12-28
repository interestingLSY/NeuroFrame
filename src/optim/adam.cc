#include "adam.h"

#include <cmath>

#include "src/basic/inference_mode.h"
#include "src/op/misc.h"

namespace NeuroFrame {

AdamOptimState::AdamOptimState(const Tensor &tensor):
	momentum(Tensor::zeros_like(tensor)),
	geo_mean(Tensor::zeros_like(tensor)) {
	cur_timestep = 0;
}

AdamOptimState::~AdamOptimState() {}

AdamOptimizer::AdamOptimizer(double beta1, double beta2, double eps):
	beta1(beta1),
	beta2(beta2),
	eps(eps) {
}

AdamOptimizer::~AdamOptimizer() {
	while (!focused_nodes.empty()) {
		remove_focus(focused_nodes.back());
	}
}

void AdamOptimizer::add_focus(const Tensor &tensor) {
	std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
	add_to_focus_list(tensor);
	node->reset_optim_state();
	node->optim_state = std::make_shared<AdamOptimState>(tensor);
}

void AdamOptimizer::remove_focus(const Tensor &tensor) {
	std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
	remove_from_focus_list(tensor);
	node->reset_optim_state();
}

void AdamOptimizer::step(double learning_rate) {
	InferenceModeGuard guard;
	guard.__enter__();
	for (Tensor &tensor : focused_nodes) {
		std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
		std::shared_ptr<AdamOptimState> optim_state = std::static_pointer_cast<AdamOptimState>(node->optim_state);
		if (!tensor.cgraph_node->grad) {
			LOG_FATAL("Gradient of one tensor is not available during gradient descent");
		}
		Tensor grad = tensor.cgraph_node->grad.value();

		optim_state->cur_timestep += 1;

		adam_grad_update(
			tensor,
			grad,
			optim_state->momentum,
			optim_state->geo_mean,
			optim_state->cur_timestep,
			learning_rate,
			beta1,
			beta2,
			eps
		);
	}
	guard.__exit__();
}

}
