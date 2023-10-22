#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"

#include "src/op/cross_entropy_loss.h"

using namespace NeuroFrame;

// Test params: <dtype, batch_size, num_classes>
class CrossEntropyLossTest : public testing::TestWithParam<std::tuple<dtype_t, int64_t, int64_t>> {
public:
};

TEST_P(CrossEntropyLossTest, CrossEntropyLossTest) {
	auto [dtype, batch_size, num_classes] = GetParam();
	Device device_h = Device::cpu();
	Device device_d = Device::cuda();

	// Here we calculate the result on both CPU and GPU, and compare them
	Tensor input_h = Tensor::randu({batch_size, num_classes}, dtype, Device::cpu());
	Tensor ground_truth_h = Tensor::randint({batch_size}, dtype_t::INT32, Device::cpu(), 0, num_classes-1);
	Tensor output_grad_h = Tensor::randu({batch_size}, dtype, Device::cpu());

	Tensor input_d = input_h.to(device_d);
	Tensor ground_truth_d = ground_truth_h.to(device_d);
	Tensor output_grad_d = output_grad_h.to(device_d);

	OpContext ctx_h;
	Tensor result_h = cross_entropy_loss_forward_manual(input_h, ground_truth_h, ctx_h);
	Tensor input_grad_h = cross_entropy_loss_backward_manual(output_grad_h, ctx_h);

	OpContext ctx_d;
	Tensor result_d = cross_entropy_loss_forward_manual(input_d, ground_truth_d, ctx_d);
	Tensor input_grad_d = cross_entropy_loss_backward_manual(output_grad_d, ctx_d);

	// Compare the results
	ASSERT_EQ(result_h, result_d.to(device_h));
	ASSERT_EQ(input_grad_h, input_grad_d.to(device_h));
}

INSTANTIATE_TEST_SUITE_P(CrossEntropyLossTest, CrossEntropyLossTest, testing::Combine(
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(1, 10, 233),
	testing::Values(1, 10, 233)
));
