#include "tensor_binary_op.h"

#include <cmath>
#include "omp.h"

#include "src/tensor/tensor.h"
#include "utils.h"
#include "src/basic/log.h"

namespace NeuroFrame::Backend::CPU {

template<typename T, typename Func>
void scalar_op_kernel(
	T* output,
	const T* input,
	const T &scalar_input,
	int64_t n,
	Func op
) {
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < n; ++i) {
		output[i] = op(input[i], scalar_input);
	}
}

#define DEFINE_TENSOR_SCALAR_OP(NAME, OP_LAMBDA) \
Tensor NAME(const Tensor &input, const Scalar &scalar) {\
	int64_t n = input.numel(); \
	Tensor output(input.shape, input.dtype, input.device); \
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, scalar_op_kernel( \
		(T*)output.data_ptr(), \
		(const T*)input.data_ptr(), \
		scalar.to_c_dtype<T>(), \
		n, \
		OP_LAMBDA \
	)); \
	return output; \
}

DEFINE_TENSOR_SCALAR_OP(tensor_adds, [](T a, T b) -> T {return a + b;})

DEFINE_TENSOR_SCALAR_OP(tensor_subs, [](T a, T b) -> T {return a - b;})

DEFINE_TENSOR_SCALAR_OP(tensor_muls, [](T a, T b) -> T {return a * b;})

DEFINE_TENSOR_SCALAR_OP(tensor_divs, [](T a, T b) -> T {return a / b;})

DEFINE_TENSOR_SCALAR_OP(tensor_pows, [](T a, T b) -> T { 
	if constexpr(std::is_same<T, half>::value) {
		return (half)std::pow((float)a, (float)b);
	} else {
		return std::pow(a, b);
	}
})


}