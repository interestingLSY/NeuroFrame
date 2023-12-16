#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"
#include "../utils.h"

#include "src/op/tensor_unary_op.h"

using namespace NeuroFrame;

enum class unary_op_type_t {
	NEGATE,
	INV,
	EXP,
	LOG
};

// Params: <unary_op_type_t, dtype_t, n>
class UnaryOpTest : public testing::TestWithParam<std::tuple<unary_op_type_t, dtype_t, int64_t>> {
public:
};

TEST_P(UnaryOpTest, UnaryOpTest) {
	auto [op_type, dtype, n] = GetParam();
	Device device_h = Device::cpu();
	Device device_d = Device::cuda();

	Tensor input_h = 
		op_type == unary_op_type_t::LOG ? Tensor::randu({n}, dtype, device_h, Scalar(0.2f), Scalar(10.0f)) :
		Tensor::randu({n}, dtype, device_h);
	Tensor input_d = input_h.to(device_d);

	OpContext ctx_h;
	Tensor output_h = 
		op_type == unary_op_type_t::NEGATE ? tensor_negate_forward_manual(input_h, ctx_h) :
		op_type == unary_op_type_t::INV ? tensor_inv_forward_manual(input_h, ctx_h) :
		op_type == unary_op_type_t::EXP ? tensor_exp_forward_manual(input_h, ctx_h) :
		op_type == unary_op_type_t::LOG ? tensor_log_forward_manual(input_h, ctx_h) :
		Tensor::zeros({n}, dtype, device_h);
	
	OpContext ctx_d;
	Tensor output_d = 
		op_type == unary_op_type_t::NEGATE ? tensor_negate_forward_manual(input_d, ctx_d) :
		op_type == unary_op_type_t::INV ? tensor_inv_forward_manual(input_d, ctx_d) :
		op_type == unary_op_type_t::EXP ? tensor_exp_forward_manual(input_d, ctx_d) :
		op_type == unary_op_type_t::LOG ? tensor_log_forward_manual(input_d, ctx_d) :
		Tensor::zeros({n}, dtype, device_d);

	ASSERT_TRUE(is_tensor_equal(output_h, output_d));
}

INSTANTIATE_TEST_SUITE_P(UnaryOpTest, UnaryOpTest, testing::Combine(
	testing::Values(unary_op_type_t::NEGATE, unary_op_type_t::INV, unary_op_type_t::EXP, unary_op_type_t::LOG),
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(2, 10, 200, 100000)
));