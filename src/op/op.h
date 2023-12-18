#pragma once

#include <functional>
#include <vector>

#include "src/tensor/tensor.h"
#include "op_utils.h"
#include "op_context.h"

namespace NeuroFrame {

typedef std::function<std::vector<Tensor>(const std::vector<Tensor>&, OpContext &, void* other_args)> op_forward_func_t;
typedef std::function<std::vector<Tensor>(const std::vector<Tensor>&, const OpContext &)> op_backward_func_t;

// perform_op: The abstraction of performing an operator.
// 
// perform_op is the "thin waist" of the whole NeuroFrame framework. It launches
// an operator with the given forward and backward functions, and the input
// tensors. In addition, it may add the op to the compute graph (for gradient
// calculation later.
// 
// The call stack looks like this:
// Used code ->
// Op wrapper (like, `relu(x)`) ->
// perform_op ->
// various kernels on various backends
// 
// The `forward` op is a lambda expression which accepts input tensors as the input
// and returns output tensors as the output. It may additionally save some intermediate
// results to OpContext for gradient calculation. 
// 
// The `backward` op is a lambda expression which accepts output gradients and 
//  the OpContext as the input and returns gradients of the input tensors
std::vector<Tensor> perform_op(
	op_forward_func_t forward_op,
	op_backward_func_t backward_op,
	std::vector<Tensor> input,
	void* other_args = nullptr
);

}
