#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"
#include "../utils.h"

#include <cstdlib>

#include "src/op/tensor_scalar_op.h"

using namespace NeuroFrame;

enum class scalar_op_type_t {
	ADDS,
	SUBS,
	MULS,
	DIVS,
	POWS
};

// Params: <scalar_op_type_t, dtype_t, n>
class ScalarOpTest : public testing::TestWithParam<std::tuple<scalar_op_type_t, dtype_t, int64_t>> {
public:
};

TEST_P(ScalarOpTest, ScalarOpTest) {
	auto [op_type, dtype, n] = GetParam();
	Device device_h = Device::cpu();
	Device device_d = Device::cuda();

	Tensor input1_h = op_type == scalar_op_type_t::POWS ? Tensor::randu({n}, dtype, device_h, 0.5f, 1.0f) : Tensor::randu({n}, dtype, device_h);
	Scalar input2 = Scalar(op_type == scalar_op_type_t::POWS ? 0.5f : 2.0f).to_dtype(dtype);
	Tensor input1_d = input1_h.to(device_d);

	OpContext ctx_h;
	Tensor output_h = 
		op_type == scalar_op_type_t::ADDS ? tensor_adds_forward_manual(input1_h, input2, ctx_h) :
		op_type == scalar_op_type_t::SUBS ? tensor_subs_forward_manual(input1_h, input2, ctx_h) :
		op_type == scalar_op_type_t::MULS ? tensor_muls_forward_manual(input1_h, input2, ctx_h) :
		op_type == scalar_op_type_t::DIVS ? tensor_divs_forward_manual(input1_h, input2, ctx_h) :
		op_type == scalar_op_type_t::POWS ? tensor_pows_forward_manual(input1_h, input2, ctx_h) :
		Tensor::zeros({n}, dtype, device_h);
	
	OpContext ctx_d;
	Tensor output_d = 
		op_type == scalar_op_type_t::ADDS ? tensor_adds_forward_manual(input1_d, input2, ctx_d) :
		op_type == scalar_op_type_t::SUBS ? tensor_subs_forward_manual(input1_d, input2, ctx_d) :
		op_type == scalar_op_type_t::MULS ? tensor_muls_forward_manual(input1_d, input2, ctx_d) :
		op_type == scalar_op_type_t::DIVS ? tensor_divs_forward_manual(input1_d, input2, ctx_d) :
		op_type == scalar_op_type_t::POWS ? tensor_pows_forward_manual(input1_d, input2, ctx_d) :
		Tensor::zeros({n}, dtype, device_d);

	ASSERT_TRUE(is_tensor_equal(output_h, output_d));
}

INSTANTIATE_TEST_SUITE_P(ScalarOpTest, ScalarOpTest, testing::Combine(
	testing::Values(scalar_op_type_t::ADDS, scalar_op_type_t::SUBS, scalar_op_type_t::MULS, scalar_op_type_t::DIVS, scalar_op_type_t::POWS),
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(2, 10, 200, 100000)
));