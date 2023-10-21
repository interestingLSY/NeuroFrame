#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"

#include "src/op/pool.h"

using namespace NeuroFrame;

// Test params: <dtype, batch_size, height, width, pool_size>
class PoolTest : public testing::TestWithParam<std::tuple<dtype_t, int64_t, int64_t, int64_t, int64_t>> {
public:
};

TEST_P(PoolTest, PoolTest) {
	auto [dtype, batch_size, input_height, input_weight, pool_size] = GetParam();
	ASSERT_EQ(input_height % pool_size, 0);
	ASSERT_EQ(input_weight % pool_size, 0);
	int64_t output_height = input_height / pool_size;
	int64_t output_weight = input_weight / pool_size;

	Device device_h = Device::cpu();
	Device device_d = Device::cuda();

	// Here we calculate the result on both CPU and GPU, and compare them
	Tensor input_h = Tensor::randu({1, batch_size, 3, input_height, input_weight}, dtype, Device::cpu());
	Tensor output_grad_h = Tensor::randu({1, batch_size, 3, output_height, output_weight}, dtype, Device::cpu());
	Tensor input_d = input_h.to(device_d);
	Tensor output_grad_d = output_grad_h.to(device_d);

	OpContext ctx_h;
	Tensor result_h = pool_forward_manual(input_h, pool_size, ctx_h);
	Tensor grad_h = pool_backward_manual(output_grad_h, ctx_h);

	OpContext ctx_d;
	Tensor result_d = pool_forward_manual(input_d, pool_size, ctx_d);
	Tensor grad_d = pool_backward_manual(output_grad_d, ctx_d);

	// Compare the results
	ASSERT_EQ(result_h, result_d.to(device_h));
	ASSERT_EQ(grad_h, grad_d.to(device_h));
}

INSTANTIATE_TEST_SUITE_P(PoolTest, PoolTest, testing::Combine(
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(1, 10, 23),
	testing::Values(12, 24, 384),
	testing::Values(12, 24, 192),
	testing::Values(2, 3, 4, 6)
));
