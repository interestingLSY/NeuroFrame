#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"

#include "src/op/relu.h"
#include "src/op/sigmoid.h"

using namespace NeuroFrame;

enum class activation_type_t {
	RELU,
	SIGMOID
};

struct ActivationTestParam {
	activation_type_t type;
	Device device;
	dtype_t dtype;
	int64_t n;
	ActivationTestParam(const std::tuple<activation_type_t, Device, dtype_t, int64_t> &param):
		type(std::get<0>(param)),
		device(std::get<1>(param)),
		dtype(std::get<2>(param)),
		n(std::get<3>(param)) {}
};

class ActivationTest : public testing::TestWithParam<std::tuple<activation_type_t, Device, dtype_t, int64_t>> {
public:
};

void activation_reference(
	activation_type_t type,
	Tensor &output,
	Tensor &input_gradient,
	const Tensor &input,
	const Tensor &output_gradient
) {
	Tensor input_h = input.to(Device::cpu());
	Tensor output_h = Tensor::zeros(input.shape, input.dtype, Device::cpu());
	Tensor input_gradient_h = Tensor::zeros(input.shape, input.dtype, Device::cpu());
	int64_t numel = input.numel();
	for (int64_t i = 0; i < numel; i++) {
		double input_x = input_h.get_elem({i}).as_scalar().as_double();
		double output_x;
		double output_gradient_x = output_gradient.get_elem({i}).as_scalar().as_double();
		double input_gradient_x;
		switch (type) {
			case activation_type_t::RELU:
				output_x = input_x > 0.0 ? input_x : 0.0;
				input_gradient_x = input_x > 0.0 ? output_gradient_x : 0.0;
				break;
			case activation_type_t::SIGMOID:
				output_x = 1.0 / (1.0 + std::exp(-input_x));
				input_gradient_x = output_x * (1.0 - output_x) * output_gradient_x;
				break;
			default:
				LOG_FATAL("Unknown activation type");
		}
		Scalar(output_x).save_to(output_h.get_elem_addr({i}), output_h.dtype);
		Scalar(input_gradient_x).save_to(input_gradient_h.get_elem_addr({i}), input_gradient_h.dtype);
	}
	output = output_h.to(input.device);
	input_gradient = input_gradient_h.to(input.device);
}

TEST_P(ActivationTest, ActivationTest) {
	ActivationTestParam param = GetParam();
	Device device = param.device;
	dtype_t dtype = param.dtype;
	activation_type_t type = param.type;
	int64_t n = param.n;

	Tensor input = Tensor::randu({n}, dtype, device);
	Tensor output_gradient = Tensor::randu({n}, dtype, device);

	Tensor reference_output = Tensor::zeros({n}, dtype, device);
	Tensor reference_input_gradient = Tensor::zeros({n}, dtype, device);
	activation_reference(type, reference_output, reference_input_gradient, input, output_gradient);

	Tensor output = Tensor::zeros({n}, dtype, device);
	Tensor input_gradient = Tensor::zeros({n}, dtype, device);
	OpContext ctx;

	switch (type) {
		case activation_type_t::RELU:
			output = relu_forward_manual(input, ctx);
			input_gradient = relu_backward_manual(output_gradient, ctx);
			break;
		case activation_type_t::SIGMOID:
			output = sigmoid_forward_manual(input, ctx);
			input_gradient = sigmoid_backward_manual(output_gradient, ctx);
			break;
	}

	ASSERT_EQ(output, reference_output);
	ASSERT_EQ(input_gradient, reference_input_gradient);
}

INSTANTIATE_TEST_SUITE_P(ReluTest, ActivationTest, testing::Combine(
	testing::Values(activation_type_t::RELU),
	testing::Values(Device::cpu(), Device::cuda(0)),
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(2, 10, 200, 100000)
));

INSTANTIATE_TEST_SUITE_P(SigmoidTest, ActivationTest, testing::Combine(
	testing::Values(activation_type_t::SIGMOID),
	testing::Values(Device::cpu(), Device::cuda(0)),
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(2, 10, 200, 100000)
));