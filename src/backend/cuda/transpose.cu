#include "transpose.h"

#include <stdexcept>
#include <cuda_runtime.h>

#include "src/tensor/tensor.h"
#include "src/basic/log.h"
#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

static constexpr int BLOCK_SIZE = 32;

// transpose_kernel: Faster transpose kernel on GPU
// This kernel utilizes shared memory to coalesce global memory access
// input: 2D tensor of shape (m, n)
// output: 2D tensor of shape (n, m)
template<typename T>
__global__ void transpose_kernel(
	const T* __restrict__ input,
	T* __restrict__ output,
	int m, int n
) {
	__shared__ T sdata[BLOCK_SIZE][BLOCK_SIZE+1];	// +1 for avoiding bank conflict

	int64_t x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int64_t y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	if (x < n && y < m) {
		sdata[threadIdx.y][threadIdx.x] = input[y * n + x];
	}

	__syncthreads();

	x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

	if (x < m && y < n) {
		output[y * m + x] = sdata[threadIdx.x][threadIdx.y];
	}
}

Tensor transpose(const Tensor &input) {
	int dim = input.dim();
	if (dim != 2) {
		LOG_FATAL("Transpose: Only support 2D tensors");
	}

	int m = input.shape[0];
	int n = input.shape[1];

	Tensor output({n, m}, input.dtype, input.device);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, transpose_kernel<<<dim3((n + BLOCK_SIZE) / BLOCK_SIZE, (m + BLOCK_SIZE) / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(
		(const T*) input.data_ptr(),
		(T*) output.data_ptr(),
		m, n
	));

	return output;
}

}