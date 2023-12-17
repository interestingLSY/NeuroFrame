#include "ops.h"

#include <pybind11/stl.h>
using namespace pybind11::literals;	// For "_a" suffix

#include "src/op/broadcast.h"
#include "src/op/convolution.h"
#include "src/op/cross_entropy_loss.h"
#include "src/op/matmul.h"
#include "src/op/pool.h"
#include "src/op/relu.h"
#include "src/op/reshape.h"
#include "src/op/sigmoid.h"
#include "src/op/tensor_binary_op.h"
#include "src/op/tensor_eq.h"
#include "src/op/tensor_reduction_op.h"
#include "src/op/tensor_scalar_op.h"
#include "src/op/tensor_unary_op.h"
#include "src/op/transpose.h"

using namespace NeuroFrame;
void init_ops(pybind11::module& m) {
	auto ops_m = m.def_submodule("ops", "NeuroFrame operators");
	
	ops_m.def("batched_convolution", &NeuroFrame::batched_convolution, "input (BCHW)"_a, "kernel (C_out, C_in, H, W)"_a);

	ops_m.def("broadcast_to", &NeuroFrame::broadcast_to, "input"_a, "target_shape"_a);

	ops_m.def("cross_entropy_loss", &NeuroFrame::cross_entropy_loss, "input"_a, "ground_truth"_a);

	ops_m.def("matmul", &NeuroFrame::matmul, "a"_a, "b"_a, "transpose_a"_a = false, "transpose_b"_a = false);

	ops_m.def("pool", &NeuroFrame::pool, "input"_a, "pool_size"_a);

	ops_m.def("relu", &NeuroFrame::relu, "input"_a);

	ops_m.def("reshape", &NeuroFrame::reshape, "input"_a, "shape"_a);

	ops_m.def("sigmoid", &NeuroFrame::sigmoid, "input"_a);

	ops_m.def("tensor_add", &NeuroFrame::tensor_add, "a"_a, "b"_a);
	ops_m.def("tensor_sub", &NeuroFrame::tensor_sub, "a"_a, "b"_a);
	ops_m.def("tensor_mul", &NeuroFrame::tensor_mul, "a"_a, "b"_a);
	ops_m.def("tensor_div", &NeuroFrame::tensor_div, "a"_a, "b"_a);
	ops_m.def("tensor_pow", &NeuroFrame::tensor_pow, "a"_a, "b"_a);

	ops_m.def("tensor_eq", &NeuroFrame::tensor_eq, "a"_a, "b"_a);

	ops_m.def("tensor_negate", &NeuroFrame::tensor_negate, "a"_a);
	ops_m.def("tensor_inv", &NeuroFrame::tensor_inv, "a"_a);
	ops_m.def("tensor_exp", &NeuroFrame::tensor_exp, "a"_a);
	ops_m.def("tensor_log", &NeuroFrame::tensor_log, "a"_a);

	ops_m.def("tensor_reduction_sum", &NeuroFrame::tensor_reduction_sum, "input"_a, "axis"_a = -1);

	#define DEFINE_TENSOR_SCALAR_OP(OP_NAME) \
	ops_m.def("tensor_"#OP_NAME, &NeuroFrame::tensor_##OP_NAME, "tensor"_a, "scalar"_a); \
	ops_m.def("tensor_"#OP_NAME, [](const Tensor &tensor, double scalar) { \
		return NeuroFrame::tensor_##OP_NAME(tensor, Scalar(scalar).to_dtype(tensor.dtype)); \
	}, "tensor"_a, "scalar"_a); \
	ops_m.def("tensor_"#OP_NAME, [](const Tensor &tensor, int64_t scalar) { \
		return NeuroFrame::tensor_##OP_NAME(tensor, Scalar(scalar).to_dtype(tensor.dtype)); \
	}, "tensor"_a, "scalar"_a);

	DEFINE_TENSOR_SCALAR_OP(adds);
	DEFINE_TENSOR_SCALAR_OP(subs);
	DEFINE_TENSOR_SCALAR_OP(muls);
	DEFINE_TENSOR_SCALAR_OP(divs);
	DEFINE_TENSOR_SCALAR_OP(pows);

	ops_m.def("transpose", [] (const NeuroFrame::Tensor &input, std::optional<int> axe1, std::optional<int> axe2) -> NeuroFrame::Tensor {
		if (axe1.has_value() != axe2.has_value()) {
			LOG_FATAL("Transpose: axe1 and axe2 must be both specified or both not specified");
		}
		if (axe1.has_value() && axe2.has_value()) {
			return NeuroFrame::transpose(input, axe1.value(), axe2.value());
		} else {
			int64_t dim = input.dim();
			if (dim < 2) {
				LOG_FATAL("Cannot transpose a tensor with dim = %ld < 2", dim);
			}
			return NeuroFrame::transpose(input, dim-2, dim-1);
		}
	}, "input"_a, "axe1"_a = std::optional<int>(), "axe2"_a = std::optional<int>());
}