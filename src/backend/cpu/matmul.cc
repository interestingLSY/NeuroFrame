#include "matmul.h"

#include <iostream>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "src/basic/log.h"


namespace NeuroFrame::Backend::CPU {

// matmul_kernel: Calculate matrix multiplication on CPU
// a is an array of shape (m, k) (if transpose_a is false) or (k, m) (if transpose_a is true)
// b is an array of shape (k, n) (if transpose_b is false) or (n, k) (if transpose_b is true)
// c is an array of shape (m, n)
template<typename T>
void matmul_kernel(
	const T* a,
	const T* b,
	T* c,
	int m, int n, int k,
	bool transpose_a,
	bool transpose_b
) {
	// typedef (std::is_same<T, double>::value ? double : float) accum_t;
	typedef std::conditional_t<std::is_same<T, double>::value, double, float> accum_t;
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			accum_t sum = (accum_t)0.0;
			for (int l = 0; l < k; l++) {
				T a_elem = transpose_a ? a[l * m + i] : a[i * k + l];
				T b_elem = transpose_b ? b[j * k + l] : b[l * n + j];
				sum += (accum_t)a_elem * (accum_t)b_elem;
			}
			c[i * n + j] = (T)sum;
		}
	}
}

Tensor matmul(const Tensor &input1, const Tensor &input2, bool transpose_a, bool transpose_b) {
	int dim1 = input1.dim();
	int dim2 = input2.dim();
	if (dim1 != 2 || dim2 != 2) {
		LOG_FATAL("Matmul: Only support 2D tensors");
	}

	int m = transpose_a ? input1.shape[1] : input1.shape[0];
	int k1 = transpose_a ? input1.shape[0] : input1.shape[1];
	int k2 = transpose_b ? input2.shape[1] : input2.shape[0];
	int n = transpose_b ? input2.shape[0] : input2.shape[1];
	if (k1 != k2) {
		LOG_FATAL("Matmul: Dimension mismatch");
	}

	Tensor output({m, n}, input1.dtype, input1.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(input1.dtype, matmul_kernel(
		(const T*)input1.data_ptr(),
		(const T*)input2.data_ptr(),
		(T*)output.data_ptr(),
		m, n, k1,
		transpose_a,
		transpose_b
	));

	return output;
}

Tensor batched_matmul(const Tensor &input1, const Tensor &input2, bool transpose_a, bool transpose_b) {
	if (input1.dim() != 3 && input1.dim() != 2) {
		LOG_FATAL("Batched matmul: the dim of input1 is not in [2, 3]");
	}
	if (input2.dim() != 3 && input2.dim() != 2) {
		LOG_FATAL("Batched matmul: the dim of input2 is not in [2, 3]");
	}

	bool is_1_batched = input1.dim() == 3;
	bool is_2_batched = input2.dim() == 3;
	if (!is_1_batched && !is_2_batched) {
		return matmul(input1, input2, transpose_a, transpose_b);
	}

	int64_t batch_count = is_1_batched ? input1.shape[0] : input2.shape[0];
	int64_t m = transpose_a ? input1.shape[is_1_batched ? 2 : 1] : input1.shape[is_1_batched ? 1 : 0];
	int64_t k1 = transpose_a ? input1.shape[is_1_batched ? 1 : 0] : input1.shape[is_1_batched ? 2 : 1];
	int64_t k2 = transpose_b ? input2.shape[is_2_batched ? 2 : 1] : input2.shape[is_2_batched ? 1 : 0];
	int64_t n = transpose_b ? input2.shape[is_2_batched ? 1 : 0] : input2.shape[is_2_batched ? 2 : 1];
	if (k1 != k2) {
		LOG_FATAL("Batched matmul: Dimension mismatch");
	}

	Tensor output({batch_count, m, n}, input1.dtype, input1.device);
	for (int i = 0; i < batch_count; ++i) {
		DISPATCH_ON_DTYPE_CPU_BACKEND(input1.dtype, matmul_kernel(
			(const T*) input1.data_ptr() + (is_1_batched ? i * m * k1 : (int64_t)0),
			(const T*) input2.data_ptr() + (is_2_batched ? i * k1 * n : (int64_t)0),
			(T*) output.data_ptr() + i * m * n,
			m, n, k1,
			transpose_a,
			transpose_b
		));
	}

	return output;
}

}