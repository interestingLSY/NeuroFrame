#include "op.h"

#include "src/basic/inference_mode.h"
#include "src/cgraph/cgraph.h"

namespace NeuroFrame {

std::vector<Tensor> perform_op(
	op_forward_func_t forward_op,
	op_backward_func_t backward_op,
	std::vector<Tensor> input,
	void* other_args
) {
	if (is_inference_mode()) {
		// Inference mode is enabled, we only need to perform forward propagation
		OpContext ctx;
		std::vector<Tensor> output = forward_op(input, ctx, other_args);
		return output;
	}
	OpContext ctx;
	std::vector<Tensor> output = forward_op(input, ctx, other_args);
	// Extend the compute graph for backward propagation
	CGraph::on_new_calculation(input, output, ctx, backward_op);
	return output;
}

}