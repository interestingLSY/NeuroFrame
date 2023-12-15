#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"
#include "../utils.h"

#include "src/op/tensor_binary_op.h"

using namespace NeuroFrame;

enum class binary_op_type_t {
	ADD,
	SUB,
	MUL,
	DIV
};

// Params: <binary_op_type_t, dtype_t, n>
class BinaryOpTest : public testing::TestWithParam<std::tuple<binary_op_type_t, dtype_t, int64_t>> {
public:
};

TEST_P(BinaryOpTest, BinaryOpTest) {
	auto [op_type, dtype, n] = GetParam();
	Device device_h = Device::cpu();
	Device device_d = Device::cuda();

	Tensor input1_h = Tensor::randu({n}, dtype, device_h);
	Tensor input2_h = Tensor::randu({n}, dtype, device_h);
	Tensor input1_d = input1_h.to(device_d);
	Tensor input2_d = input2_h.to(device_d);

	OpContext ctx_h;
	Tensor output_h = 
		op_type == binary_op_type_t::ADD ? tensor_add_forward_manual(input1_h, input2_h, ctx_h) :
		op_type == binary_op_type_t::SUB ? tensor_sub_forward_manual(input1_h, input2_h, ctx_h) :
		op_type == binary_op_type_t::MUL ? tensor_mul_forward_manual(input1_h, input2_h, ctx_h) :
		op_type == binary_op_type_t::DIV ? tensor_div_forward_manual(input1_h, input2_h, ctx_h) :
		Tensor::zeros({n}, dtype, device_h);
	
	OpContext ctx_d;
	Tensor output_d = 
		op_type == binary_op_type_t::ADD ? tensor_add_forward_manual(input1_d, input2_d, ctx_d) :
		op_type == binary_op_type_t::SUB ? tensor_sub_forward_manual(input1_d, input2_d, ctx_d) :
		op_type == binary_op_type_t::MUL ? tensor_mul_forward_manual(input1_d, input2_d, ctx_d) :
		op_type == binary_op_type_t::DIV ? tensor_div_forward_manual(input1_d, input2_d, ctx_d) :
		Tensor::zeros({n}, dtype, device_d);

	ASSERT_TRUE(is_tensor_equal(output_h, output_d));
}

INSTANTIATE_TEST_SUITE_P(BinaryOpTest, BinaryOpTest, testing::Combine(
	testing::Values(binary_op_type_t::ADD, binary_op_type_t::SUB, binary_op_type_t::MUL, binary_op_type_t::DIV),
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(2, 10, 200, 100000)
));