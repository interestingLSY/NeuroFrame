#include "adam.h"

#include <cmath>

#include "src/basic/inference_mode.h"
#include "src/op/tensor_binary_op.h"
#include "src/op/tensor_scalar_op.h"

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
	for (const Tensor &tensor : focused_nodes) {
		remove_focus(tensor);
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

		optim_state->momentum = tensor_add(
			tensor_muls(
				optim_state->momentum,
				Scalar(beta1, tensor.dtype)
			),
			tensor_muls(
				grad,
				Scalar(1.0 - beta1, tensor.dtype)
			)
		);

		optim_state->geo_mean = tensor_add(
			tensor_muls(
				optim_state->geo_mean,
				Scalar(beta2, tensor.dtype)
			),
			tensor_muls(
				tensor_mul(
					grad,
					grad
				),
				Scalar(1.0 - beta2, tensor.dtype)
			)
		);

		Tensor adjusted_momentum = tensor_divs(
			optim_state->momentum,
			Scalar(1.0 - std::pow(beta1, optim_state->cur_timestep), tensor.dtype)
		);
		Tensor adjusted_geo_mean = tensor_divs(
			optim_state->geo_mean,
			Scalar(1.0 - std::pow(beta2, optim_state->cur_timestep), tensor.dtype)
		);

		Tensor new_weight = tensor_sub(
			tensor,
			tensor_muls(
				tensor_div(
					adjusted_momentum,
					tensor_adds(
						tensor_pows(
							adjusted_geo_mean,
							Scalar(0.5, tensor.dtype)
						),
						Scalar(eps, tensor.dtype)
					)
				),
				Scalar(learning_rate, tensor.dtype)
			)
		);
		tensor.mem_frag.copy_from(new_weight.mem_frag);
	}
	guard.__exit__();
}

}
