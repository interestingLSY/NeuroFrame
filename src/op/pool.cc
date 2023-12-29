#include "pool.h"

#include <cassert>

#include "src/backend/cpu/pool.h"
#include "src/backend/cuda/pool.h"

namespace NeuroFrame {

// pool_forward_func: The forward function of pool operator.
// Input:
//	- input: The input tensor, (batch_size, c, height, weight)
// Output:
//	- result: The result tensor
// SavedContext:
//	- saved_tensors[0]: The input
//	- saved_tensors[1]: The output
// OtherArgs: 8byte
//	- other_args[0]: PoolForwardArgs

struct PoolForwardArgs {
	int64_t pool_size;
	int64_t stride;
	int64_t padding;
};

static op_forward_func_t pool_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);

	assert(other_args);
	PoolForwardArgs args = *(PoolForwardArgs*)other_args;
	ctx.save_args(other_args, sizeof(PoolForwardArgs));

	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		pool_forward(input[0], args.pool_size, args.stride, args.padding)
	);
	ctx.save_for_backward(input[0]);
	ctx.save_for_backward(result);

	return {result};
};

static op_backward_func_t pool_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	// do_basic_checkings_in_forward_and_backward(output_grad, ctx);	Do not check this since max_mask is always in INT8
	
	PoolForwardArgs args = *(PoolForwardArgs*)ctx.get_saved_args();
	Tensor input = ctx.get_saved_tensors()[0];
	Tensor output = ctx.get_saved_tensors()[1];

	Tensor input_grad = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		pool_backward(
			output_grad[0], 
			input,
			output,
			args.pool_size,
			args.stride,
			args.padding
		)
	);

	return {input_grad};
};

Tensor pool_forward_manual(const Tensor &input, int64_t pool_size, int64_t stride, int64_t padding, OpContext &ctx) {
	PoolForwardArgs args = {pool_size, stride, padding};
	return pool_forward_func({input}, ctx, &args)[0];
}

Tensor pool_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return pool_backward_func({output_grad}, ctx)[0];
}

Tensor pool(const Tensor &input, int64_t pool_size, int64_t stride, int64_t padding) {
	PoolForwardArgs args = {pool_size, stride, padding};
	return perform_op(pool_forward_func, pool_backward_func, {input}, &args)[0];
}

}
