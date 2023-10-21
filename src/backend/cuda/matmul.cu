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
			1LL*n*k,
			(T*) Aarray.data_ptr(),
			get_cuda_datatype<T>(),
			transpose_a ? m : k,
			1LL*k*m,
			&beta,
			(T*) Carray.data_ptr(),
			get_cuda_datatype<T>(),
			n,
			1LL*n*m,
			batch_count,
			get_cublas_compute_type<T>(),
			CUBLAS_GEMM_DEFAULT_TENSOR_OP
		);
		if (status != CUBLAS_STATUS_SUCCESS) {
			LOG_FATAL("CublasWrapper::gemmStridedBatched failed: %d", status);
		}
	}());
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

}