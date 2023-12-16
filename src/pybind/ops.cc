#include "ops.h"

#include <pybind11/stl.h>
using namespace pybind11::literals;	// For "_a" suffix

#include "src/op/convolution.h"
#include "src/op/cross_entropy_loss.h"
#include "src/op/matmul.h"
#include "src/op/pool.h"
#include "src/op/relu.h"
#include "src/op/reshape.h"
#include "src/op/sigmoid.h"
#include "src/op/tensor_binary_op.h"
#include "src/op/tensor_eq.h"
#include "src/op/tensor_unary_op.h"
#include "src/op/transpose.h"

void init_ops(pybind11::module& m) {
	auto ops_m = m.def_submodule("ops", "NeuroFrame operators");
	
	ops_m.def("batched_convolution", &NeuroFrame::batched_convolution, "input (BCHW)"_a, "kernel (C_out, C_in, H, W)"_a);

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

	ops_m.def("transpose", &NeuroFrame::transpose, "input"_a);
}