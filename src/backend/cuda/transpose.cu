#include "transpose.h"

#include <stdexcept>
#include <cuda_runtime.h>

#include "src/tensor/tensor.h"
#include "src/basic/log.h"
#include "utils.h"
#include "src/utils/utils.h"

namespace NeuroFrame::Backend::CUDA {

// transpose_kernel: Transpose on GPU
// We view the tensor `input` as [d1, d2, d3, d4, d5] and swap the 2-nd (d2) and 4-th (d4) axes.
// The output tensor has shape [d1, d4, d3, d2, d5].
// grid size: [min(d1, GRID_SIDE_LEN), min(d2, GRID_SIDE_LEN), min(d3, GRID_SIDE_LEN)]
// block size: [min(d4, BLOCK_SIDE_LEN), min(d5, BLOCK_SIDE_LEN)]
static constexpr int64_t GRID_SIDE_LEN = 16;
static constexpr int64_t BLOCK_SIDE_LEN = 32;
template<typename T>
__global__ void transpose_kernel(
	const T* __restrict__ input,
	T* __restrict__ output,
	int64_t d1,
	int64_t d2,
	int64_t d3,
	int64_t d4,
	int64_t d5
) {
	for (int64_t a1 = blockIdx.x; a1 < d1; a1 += GRID_SIDE_LEN) {
		for (int64_t a2 = blockIdx.y; a2 < d2; a2 += GRID_SIDE_LEN) {
			for (int64_t a3 = blockIdx.z; a3 < d3; a3 += GRID_SIDE_LEN) {
				for (int64_t a4 = threadIdx.x; a4 < d4; a4 += BLOCK_SIDE_LEN) {
					for (int64_t a5 = threadIdx.y; a5 < d5; a5 += BLOCK_SIDE_LEN) {
						int64_t input_index = INDEX_5D(
							d1, d2, d3, d4, d5,
							a1, a2, a3, a4, a5
						);
						int64_t output_index = INDEX_5D(
							d1, d4, d3, d2, d5,
							a1, a4, a3, a2, a5
						);
						output[output_index] = input[input_index];
					}
				}
			}
		}
	}
}

Tensor transpose(const Tensor &input, int axe1, int axe2) {
	int dim = input.dim();
	std::vector<int64_t> shape = input.shape;
	if (axe1 < 0 || axe1 >= dim || axe2 < 0 || axe2 >= dim) {
		LOG_FATAL("Transpose: Invalid axes");
	}
	if (axe1 == axe2) {
		return input;
	}
	if (axe1 > axe2) {
		std::swap(axe1, axe2);
	}

	int64_t d1 = get_product_over_vector(shape, 0, axe1);
	int64_t d2 = shape[axe1];
	int64_t d3 = get_product_over_vector(shape, axe1 + 1, axe2);
	int64_t d4 = shape[axe2];
	int64_t d5 = get_product_over_vector(shape, axe2 + 1);

	std::vector<int64_t> new_shape = shape;
	std::swap(new_shape[axe1], new_shape[axe2]);

	Tensor output(new_shape, input.dtype, input.device);
	dim3 grid_dim(
		std::min(d1, GRID_SIDE_LEN),
		std::min(d2, GRID_SIDE_LEN),
		std::min(d3, GRID_SIDE_LEN)
	);
	dim3 block_dim(
		std::min(d4, BLOCK_SIDE_LEN),
		std::min(d5, BLOCK_SIDE_LEN)
	);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, transpose_kernel<<<grid_dim, block_dim>>>(
		(const T*) input.data_ptr(),
		(T*) output.data_ptr(),
		d1, d2, d3, d4, d5
	));

	return output;
}

}