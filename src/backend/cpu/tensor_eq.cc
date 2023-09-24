#include "tensor_eq.h"

#include <cuda_fp16.h>

#include "utils.h"
#include "src/backend/utils.h"	// For HALF_ABS_THRES, HALF_REL_THRES, etc.

namespace NeuroFrame::Backend::CPU {

template<typename T>
bool is_elem_eq(const T &a, const T &b) {
	T abs_err = std::abs(a - b);
	T rel_err = abs_err / (std::max(std::abs(a), std::abs(b)) + (T)(1e-5));
	if constexpr (std::is_same_v<T, half>) {
		return abs_err <= HALF_ABS_THRES && rel_err <= HALF_REL_THRES;
	} else if constexpr (std::is_same_v<T, float>) {
		return abs_err <= FLOAT_ABS_THRES && rel_err <= FLOAT_REL_THRES;
	} else if constexpr (std::is_same_v<T, double>) {
		return abs_err <= DOUBLE_ABS_THRES && rel_err <= DOUBLE_REL_THRES;
	} else {
		return a == b;
	}
} 

template<typename T>
bool tensor_eq_kernel(
	const T* arr1,
	const T* arr2,
	int64_t n_elems
) {
	for (int64_t i = 0; i < n_elems; i++) {
		if (!is_elem_eq(arr1[i], arr2[i])) {
			return false;
		}
	}
	return true;
}

bool tensor_eq(const Tensor &input1, const Tensor &input2) {
	// The two tensors are guaranteed to have the same shape, data type, and device
	bool result = DISPATCH_ON_DTYPE_CPU_BACKEND(input1.dtype, tensor_eq_kernel(
		(const T*)input1.data_ptr(),
		(const T*)input2.data_ptr(),
		input1.numel()
	));
	return result;
}

}