#include "broadcast.h"

#include <cassert>
#include <stdexcept>

#include "src/utils/utils.h"
#include "src/backend/cpu/broadcast.h"
#include "src/backend/cuda/broadcast.h"

namespace NeuroFrame {

// broadcast_to: Tensor broadcasting
// Input:
//	- input: The input Tensor
//	- shape: The new shape of the Tensor
// Output:
//	- result: The result Tensor
// SavedContext:
// 	- input (indeed we only need the shape of the input Tensor)
// other_args: None

static op_forward_func_t broadcast_to_forward_func = [](const std::vector<Tensor> &inputs, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(inputs, ctx);
	assert(other_args);

	Tensor input = inputs[0];
	std::vector<int64_t> shape = *(std::vector<int64_t>*)other_args;
	if ((int64_t)shape.size() < input.dim()) {
		LOG_FATAL("The number of dimensions of the new shape (%s, %lu) must >="
				  "equal to the number of dimensions of the input Tensor (%s, %ld)",
				  vec_to_string(shape).c_str(), shape.size(),
				  vec_to_string(input.shape).c_str(), input.dim());
	}

	int64_t num_prepended_dims = shape.size() - input.dim();
	for (size_t i = num_prepended_dims; i < shape.size(); ++i) {
		if (shape[i]%input.shape[i-num_prepended_dims] != 0) {
			LOG_FATAL("The new shape (%s) must be a multiple of the old shape (%s)",
					  vec_to_string(shape).c_str(),
					  vec_to_string(input.shape).c_str());
		}
	}
	Tensor result = DISPATCH_TO_BACKEND(
		input.device.type,
		broadcast_to(input, shape)
	);
	ctx.save_for_backward(input);
	return {result};
};

static op_backward_func_t broadcast_to_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	std::vector<int64_t> old_shape = ctx.get_saved_tensor(0).shape;
	Tensor result = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		broadcast_to_backward(output_grad[0], old_shape)
	);
	return {result};
};

Tensor broadcast_to_forward_manual(const Tensor &input, OpContext &ctx, const std::vector<int64_t> &shape) {
	return broadcast_to_forward_func({input}, ctx, (void*)&shape)[0];
}

Tensor broadcast_to_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return broadcast_to_backward_func({output_grad}, ctx)[0];
}

Tensor broadcast_to(const Tensor &input, const std::vector<int64_t> &shape) {
	return perform_op(broadcast_to_forward_func, broadcast_to_backward_func, {input}, (void*)(&shape))[0];
}

}
