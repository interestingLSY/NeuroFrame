#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"

#include "src/op/convolution.h"

using namespace NeuroFrame;

// Test params: <dtype, batch_size, c_in, height, width, c_out, kh, kw>
class ConvolutionTest : public testing::TestWithParam<std::tuple<dtype_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> {
public:
};

TEST_P(ConvolutionTest, ConvolutionTest) {
	auto [dtype, batch_size, c_in, h, w, c_out, kh, kw] = GetParam();
	Device device_h = Device::cpu();
	Device device_d = Device::cuda();

	// Here we calculate the result on both CPU and GPU, and compare them
	Tensor input_image_h = Tensor::randu({batch_size, c_in, h, w}, dtype, Device::cpu());
	Tensor kernel_h = Tensor::randu({c_out, c_in, kh, kw}, dtype, Device::cpu());
	Tensor output_grad_h = Tensor::randu({batch_size, c_out, h, w}, dtype, Device::cpu());

	Tensor input_image_d = input_image_h.to(device_d);
	Tensor kernel_d = kernel_h.to(device_d);
	Tensor output_grad_d = output_grad_h.to(device_d);

	OpContext ctx_h;
	Tensor result_h = batched_convolution_forward_manual(input_image_h, kernel_h, ctx_h);
	std::vector<Tensor> grads_h = batched_convolution_backward_manual(output_grad_h, ctx_h);
	Tensor input_image_grad_h = grads_h[0];
	Tensor kernel_grad_h = grads_h[1];

	OpContext ctx_d;
	Tensor result_d = batched_convolution_forward_manual(input_image_d, kernel_d, ctx_d);
	std::vector<Tensor> grads_d = batched_convolution_backward_manual(output_grad_d, ctx_d);
	Tensor input_image_grad_d = grads_d[0];
	Tensor kernel_grad_d = grads_d[1];

	// Compare the results
	ASSERT_EQ(result_h, result_d.to(device_h));
	ASSERT_EQ(input_image_grad_h, input_image_grad_d.to(device_h));
	ASSERT_EQ(kernel_grad_h, kernel_grad_d.to(device_h));
}

INSTANTIATE_TEST_SUITE_P(ConvolutionTest, ConvolutionTest, testing::Combine(
	// testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(1, 23),
	testing::Values(1, 3, 15),
	testing::Values(5, 10, 128),
	testing::Values(5, 10, 128),
	testing::Values(1, 3, 15),
	testing::Values(3, 5),
	testing::Values(3, 5)
));
