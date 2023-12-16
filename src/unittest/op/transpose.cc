#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "src/basic/scalar.h"
#include "src/basic/device.h"
#include "src/basic/log.h"

#include "src/tensor/tensor.h"
#include "../utils.h"

#include "src/op/transpose.h"

using namespace NeuroFrame;

// Test params: <device, dtype, m, n>
class TransposeTest : public testing::TestWithParam<std::tuple<dtype_t>> {
public:
};

TEST_P(TransposeTest, TransposeTest) {
	auto [dtype] = GetParam();

	std::mt19937 rng(0);
	for (int _ = 0; _ < 100; ++_) {
		int dim = rng() % 6 + 1;
		std::vector<int64_t> shape(dim);
		for (int i = 0; i < dim; ++i)
			shape[i] = rng() % 10 + 1;
		int axe1 = rng() % dim;
		int axe2 = rng() % dim;

		Tensor input_h = Tensor::randu(shape, dtype, Device::cpu());
		Tensor input_d = input_h.cuda();

		Tensor output_h = transpose(input_h, axe1, axe2);
		Tensor output_d = transpose(input_d, axe1, axe2);
		
		ASSERT_TRUE(is_tensor_equal(output_h, output_d));
	}
}

INSTANTIATE_TEST_SUITE_P(TransposeTest, TransposeTest, testing::Combine(
	testing::Values(dtype_t::FLOAT16, dtype_t::FLOAT32, dtype_t::FLOAT64)
));