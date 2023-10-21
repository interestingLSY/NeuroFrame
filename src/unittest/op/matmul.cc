#include <cmath>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"

#include "src/op/matmul.h"

using namespace NeuroFrame;

// Test params: <dtype, m, n, k, transpose_a, transpose_b>
class MatmulTest : public testing::TestWithParam<std::tuple<dtype_t, int64_t, int64_t, int64_t>> {
public:
};

TEST_P(MatmulTest, MatmulTest) {
	auto [dtype, m, n, k] = GetParam();
	Device device_h = Device::cpu();
	Device device_d = Device::cuda();

	// Here we calculate the result on both CPU and GPU, and compare them
	Tensor a_h = Tensor::randu({m, k}, dtype, Device::cpu());
	Tensor b_h = Tensor::randu({k, n}, dtype, Device::cpu());
	Tensor output_grad_h = Tensor::randu({m, n}, dtype, Device::cpu());

	OpContext ctx_h;
	Tensor result_h = matmul_forward_manual(a_h, b_h, ctx_h);
	std::vector<Tensor> grad_h = matmul_backward_manual(output_grad_h, ctx_h);

	OpContext ctx_d;
	Tensor result_d = matmul_forward_manual(a_h.to(device_d), b_h.to(device_d), ctx_d);
	// std::vector<Tensor> grad_d = matmul_backward_manual(output_grad_h.to(device_d), ctx_d);

	// Compare the results
	ASSERT_EQ(result_h, result_d.to(device_h));
}

INSTANTIATE_TEST_SUITE_P(MatmulTest, MatmulTest, testing::Combine(
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64),
	testing::Values(2, 10, 200, 1145),
	testing::Values(2, 10, 200, 1145),
	testing::Values(2, 10, 200, 114)
));