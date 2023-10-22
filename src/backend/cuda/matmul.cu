#include "matmul.h"

#include <iostream>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "src/basic/log.h"


namespace NeuroFrame::Backend::CUDA {

namespace CublasWrapper{
cublasHandle_t handle;

#define checkCublasStatus(status) checkCublasStatus_line((status), __FILE__, __LINE__)

// A tiny class to create and destroy cublas handle
class CublasHandleCreator {
public:
	CublasHandleCreator() {
		cublasCreate(&handle);
	}
	~CublasHandleCreator() {
		cublasDestroy(handle);
	}
} _cublas_handle_creator;

template<typename T>
cublasComputeType_t get_cublas_compute_type() {
	if (std::is_same<T, half>::value) {
		return CUBLAS_COMPUTE_16F;
	} else if (std::is_same<T, float>::value) {
		return CUBLAS_COMPUTE_32F;
	} else if (std::is_same<T, double>::value) {
		return CUBLAS_COMPUTE_64F;
	} else {
		LOG_FATAL("Cublas compute type: Unsupported type");
	}
}

static void cublas_gemm_strided_batched(
	int m,
	int n,
	int k,
	const Tensor& Aarray,
	const Tensor& Barray,
	Tensor& Carray,
	int batch_count,
	bool transpose_a,
	bool transpose_b,
	int64_t stride_a,
	int64_t stride_b,
	int64_t stride_c
) {
	DISPATCH_ON_DTYPE_CUDA_BACKEND(Aarray.dtype, [&]() {
		const T alpha = (T)1.0;
		const T beta = (T)0.0;
		cublasStatus_t status = cublasGemmStridedBatchedEx(
			handle,
			transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
			transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
			n,
			m,
			k,
			&alpha,
			(T*) Barray.data_ptr(),
			get_cuda_datatype<T>(),
			transpose_b ? k : n,
			stride_b,
			(T*) Aarray.data_ptr(),
			get_cuda_datatype<T>(),
			transpose_a ? m : k,
			stride_a,
			&beta,
			(T*) Carray.data_ptr(),
			get_cuda_datatype<T>(),
			n,
			stride_c,
			batch_count,
			get_cublas_compute_type<T>(),
			CUBLAS_GEMM_DEFAULT_TENSOR_OP
		);
		if (status != CUBLAS_STATUS_SUCCESS) {
			LOG_FATAL("CublasWrapper::gemmStridedBatched failed: %d", status);
		}
	}());
}

// cublas_gemm_batched: Calculate C[i] = A[i] x B[i], where A, B and C are
// arrays of matrices.
// Although it accepts tensors as input, it does not care about its shape. It
// only treats them as a bunch of numbers, i.e. only data_ptr and numel are used.
static void cublas_gemm_batched(
	int m,
	int n,
	int k,
	const Tensor& Aarray,
	const Tensor& Barray,
	Tensor& Carray,
	int batch_count,
	bool transpose_a,
	bool transpose_b
) {
	cublas_gemm_strided_batched(
		m, n, k,
		Aarray, Barray, Carray,
		batch_count,
		transpose_a, transpose_b,
		1LL*m*k, 1LL*n*k, 1LL*n*m
	);
}

// cublas_gemm: Calculate C = A x B
// Although it accepts tensors as input, it does not care about its shape. It
// only treats them as a bunch of numbers, i.e. only data_ptr and numel are used.
static void cublas_gemm(
	int m,
	int n,
	int k,
	const Tensor& Aarray,
	const Tensor& Barray,
	Tensor& Carray,
	bool transpose_a,
	bool transpose_b
) {
	cublas_gemm_batched(
		m,
		n,
		k,
		Aarray,
		Barray,
		Carray,
		1,
		transpose_a,
		transpose_b
	);
}

}	// namespace CublasWrapper

// Matmul: matrix multiplication.
// Now it only support the case where input1 and input2 are both 2D tensors.
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
	CublasWrapper::cublas_gemm(
		m,
		n,
		k1,
		input1,
		input2,
		output,
		transpose_a,
		transpose_b
	);

	return output;
}

Tensor batched_matmul(const Tensor &input1, const Tensor &input2, bool transpose_a, bool transpose_b) {
	if (input1.dim() != 3 && input1.dim() != 2) {
		LOG_FATAL("Batched matmul: the dim of input1 is not in [2, 3]");
	}
	if (input2.dim() != 3 && input2.dim() != 2) {
		LOG_FATAL("Batched matmul: the dim of input2 is not in [2, 3]");
	}

	if (input1.dim() == 2 && input2.dim() == 2) {
		return matmul(input1, input2, transpose_a, transpose_b);
	} else if (input1.dim() == 2 && input2.dim() == 3) {
		int batch_count = input2.shape[0];
		int m = transpose_a ? input1.shape[1] : input1.shape[0];
		int k1 = transpose_a ? input1.shape[0] : input1.shape[1];
		int k2 = transpose_b ? input2.shape[2] : input2.shape[1];
		int n = transpose_b ? input2.shape[1] : input2.shape[2];
		if (k1 != k2) {
			LOG_FATAL("Batched matmul: Dimension mismatch");
		}

		Tensor output({batch_count, m, n}, input1.dtype, input1.device);
		CublasWrapper::cublas_gemm_strided_batched(
			m,
			n,
			k1,
			input1,
			input2,
			output,
			batch_count,
			transpose_a,
			transpose_b,
			0LL, 1LL*k1*n, 1LL*m*n
		);

		return output;
	} else if (input1.dim() == 3 && input2.dim() == 2) {
		int batch_count = input1.shape[0];
		int m = transpose_a ? input1.shape[2] : input1.shape[1];
		int k1 = transpose_a ? input1.shape[1] : input1.shape[2];
		int k2 = transpose_b ? input2.shape[1] : input2.shape[0];
		int n = transpose_b ? input2.shape[0] : input2.shape[1];
		if (k1 != k2) {
			LOG_FATAL("Batched matmul: Dimension mismatch");
		}

		Tensor output({batch_count, m, n}, input1.dtype, input1.device);
		CublasWrapper::cublas_gemm_strided_batched(
			m,
			n,
			k1,
			input1,
			input2,
			output,
			batch_count,
			transpose_a,
			transpose_b,
			1LL*m*k1, 0LL, 1LL*m*n
		);

		return output;
	} else {
		int batch_count = input1.shape[0];
		int m = transpose_a ? input1.shape[2] : input1.shape[1];
		int k1 = transpose_a ? input1.shape[1] : input1.shape[2];
		int k2 = transpose_b ? input2.shape[2] : input2.shape[1];
		int n = transpose_b ? input2.shape[1] : input2.shape[2];
		if (k1 != k2) {
			LOG_FATAL("Batched matmul: Dimension mismatch");
		}

		Tensor output({batch_count, m, n}, input1.dtype, input1.device);
		CublasWrapper::cublas_gemm_batched(
			m,
			n,
			k1,
			input1,
			input2,
			output,
			batch_count,
			transpose_a,
			transpose_b
		);

		return output;
	}
}

}