#include "broadcast.h"

#include <cassert>
#include <stdexcept>

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

static op_forward_func_t broadcast_to_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	assert(other_args);
	std::vector<int64_t> shape = *(std::vector<int64_t>*)other_args;
	if ((int64_t)shape.size() != input[0].dim()) {
		LOG_FATAL("The number of dimensions of the new shape must be equal to the number of dimensions of the input Tensor");
	}
	for (size_t i = 0; i < shape.size(); ++i) {
		if (shape[i]%input[0].shape[i] != 0) {
			LOG_FATAL("The new shape must be a multiple of the old shape");
		}
	}
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		broadcast_to(input[0], shape)
	);
	ctx.save_for_backward(input[0]);
	return {result};
};

static op_backward_func_t broadcast_to_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	std::vector<int64_t> old_shape = ctx.get_saved_tensor(0).shape;
	// TODO
	throw std::runtime_error("Not implemented");
	Tensor result = output_grad[0].reshape(old_shape);
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
