#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"

#include "src/op/transpose.h"

using namespace NeuroFrame;

// Test params: <device, dtype, m, n>
class TransposeTest : public testing::TestWithParam<std::tuple<Device, dtype_t, int64_t, int64_t>> {
public:
};

void transpose_reference(
	Tensor &output,
	Tensor &input_gradient,
	const Tensor &input,
	const Tensor &output_gradient
) {
	int64_t m = input.shape[0];
	int64_t n = input.shape[1];

	Tensor input_h = input.to(Device::cpu());
	Tensor output_h = Tensor::zeros({n, m}, input.dtype, Device::cpu());
	Tensor input_gradient_h = Tensor::zeros({m, n}, input.dtype, Device::cpu());
	
	assert (input.dim() == 2);
	for (int64_t i = 0; i < m; i++) {
		for (int64_t j = 0; j < n; j++) {
			output_h.get_elem({j, i}) = input_h.get_elem({i, j}).as_scalar();
			input_gradient_h.get_elem({i, j}) = output_gradient.get_elem({j, i}).as_scalar();
		}
	}

	output = output_h.to(input.device);
	input_gradient = input_gradient_h.to(input.device);
}

TEST_P(TransposeTest, TransposeTest) {
	auto [device, dtype, m, n] = GetParam();

	Tensor input = Tensor::randu({m, n}, dtype, device);
	Tensor output_gradient = Tensor::randu({n, m}, dtype, device);

	Tensor reference_output = Tensor::zeros({n, m}, dtype, device);
	Tensor reference_input_gradient = Tensor::zeros({m, n}, dtype, device);
	transpose_reference(reference_output, reference_input_gradient, input, output_gradient);

	OpContext ctx;
	Tensor output = transpose_forward_manual(input, ctx);
	Tensor input_gradient = transpose_backward_manual(output_gradient, ctx);

	ASSERT_EQ(output, reference_output);
	ASSERT_EQ(input_gradient, reference_input_gradient);
}

INSTANTIATE_TEST_SUITE_P(TransposeTest, TransposeTest, testing::Combine(
	testing::Values(Device::cpu(), Device::cuda(0)),
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(2, 10, 200, 1145),
	testing::Values(2, 10, 200)
));