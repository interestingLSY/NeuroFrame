#include "transpose.h"

#include <stdexcept>

#include "omp.h"

#include "src/tensor/tensor.h"
#include "utils.h"
#include "src/basic/log.h"

namespace NeuroFrame::Backend::CPU {

template<typename T>
void transpose_kernel(
	const T* input,
	T* output,
	int m, int n
) {
	#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j ++) {
			output[j * m + i] = input[i * n + j];
		}
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
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, transpose_kernel(
		(const T*)input.data_ptr(),
		(T*)output.data_ptr(),
		m, n
	));
	
	return output;
}

}