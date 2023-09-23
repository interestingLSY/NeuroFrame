#include "op.h"

namespace NeuroFrame {

std::vector<Tensor> perform_op(
	op_forward_func_t forward_op,
	op_backward_func_t backward_op,
	const std::vector<Tensor> &input
) {
	OpContext ctx;
	std::vector<Tensor> output = forward_op(input, ctx);
	// TODO Extend the compute graph for backward propagation
	return output;
}

}