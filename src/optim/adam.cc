#include "adam.h"

#include "src/basic/inference_mode.h"
#include "src/op/tensor_binary_op.h"
#include "src/op/tensor_scalar_op.h"

namespace NeuroFrame {

AdamOptimState::AdamOptimState(const Tensor &tensor):
	momentum(Tensor::zeros_like(tensor)),
	geo_mean(Tensor::zeros_like(tensor)) {
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
		OpContext temp_ctx;

		optim_state->momentum = tensor_add_forward_manual(
			tensor_muls_forward_manual(
				optim_state->momentum,
				Scalar(beta1, tensor.dtype),
				temp_ctx
			),
			tensor_muls_forward_manual(
				grad,
				Scalar(1.0 - beta1, tensor.dtype),
				temp_ctx
			),
			temp_ctx
		);

		optim_state->geo_mean = tensor_add_forward_manual(
			tensor_muls_forward_manual(
				optim_state->geo_mean,
				Scalar(beta2, tensor.dtype),
				temp_ctx
			),
			tensor_muls_forward_manual(
				tensor_mul_forward_manual(
					grad,
					grad,
					temp_ctx
				),
				Scalar(1.0 - beta2, tensor.dtype),
				temp_ctx
			),
			temp_ctx
		);

		Tensor new_weight = tensor_sub_forward_manual(
			tensor,
			tensor_muls_forward_manual(
				tensor_div_forward_manual(
					optim_state->momentum,
					tensor_adds_forward_manual(
						tensor_pows_forward_manual(
							optim_state->geo_mean,
							Scalar(0.5, tensor.dtype),
							temp_ctx
						),
						Scalar(eps, tensor.dtype),
						temp_ctx
					),
					temp_ctx
				),
				Scalar(learning_rate, tensor.dtype),
				temp_ctx
			),
			temp_ctx
		);
		tensor.mem_frag.copy_from(new_weight.mem_frag);
	}
	guard.__exit__();
}

}
