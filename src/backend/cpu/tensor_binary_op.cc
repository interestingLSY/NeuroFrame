#include "tensor_binary_op.h"

#include <cmath>
#include "omp.h"

#include "src/tensor/tensor.h"
#include "utils.h"
#include "src/basic/log.h"

namespace NeuroFrame::Backend::CPU {

template<typename T, typename Func>
void binary_op_kernel(
	T* output,
	const T* input1,
	const T* input2,
	int64_t n,
	Func op
) {
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < n; ++i) {
		output[i] = op(input1[i], input2[i]);
	}
}

#define DEFINE_BINARY_OP(name, OP_LAMBDA) \
Tensor name(const Tensor &input1, const Tensor &input2) {\
	int64_t n = input1.numel(); \
	Tensor output(input1.shape, input1.dtype, input1.device); \
	DISPATCH_ON_DTYPE_CPU_BACKEND(input1.dtype, binary_op_kernel( \
		(T*)output.data_ptr(), \
		(const T*)input1.data_ptr(), \
		(const T*)input2.data_ptr(), \
		n, \
		OP_LAMBDA \
	)); \
	return output; \
}

DEFINE_BINARY_OP(tensor_add, [](T a, T b) -> T { return a + b; })

DEFINE_BINARY_OP(tensor_sub, [](T a, T b) -> T { return a - b; })

DEFINE_BINARY_OP(tensor_mul, [](T a, T b) -> T { return a * b; })

DEFINE_BINARY_OP(tensor_div, [](T a, T b) -> T { return a / b; })

DEFINE_BINARY_OP(tensor_pow, [](T a, T b) -> T { 
	if constexpr(std::is_same<T, half>::value) {
		return (half)std::pow((float)a, (float)b);
	} else {
		return std::pow(a, b);
	}
})

}