#include "tensor_binary_op.h"

#include <cmath>
#include "omp.h"

#include "src/tensor/tensor.h"
#include "src/basic/log.h"

#include "utils.h"
#include "../utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T, typename Func>
void unary_op_kernel(
	T* output,
	const T* input,
	int64_t n,
	Func op
) {
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < n; ++i) {
		output[i] = op(input[i]);
	}
}

#define DEFINE_UNARY_OP(name, OP_LAMBDA) \
Tensor name(const Tensor &input) {\
	int64_t n = input.numel(); \
	Tensor output(input.shape, input.dtype, input.device); \
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, unary_op_kernel( \
		(T*)output.data_ptr(), \
		(const T*)input.data_ptr(), \
		n, \
		OP_LAMBDA \
	)); \
	return output; \
}

DEFINE_UNARY_OP(tensor_negate, [](T a) -> T { return -a; })

DEFINE_UNARY_OP(tensor_inv, [](T a) -> T { return (T)1.0/a; })

DEFINE_UNARY_OP(tensor_exp, [](T a) -> T {
	if constexpr(std::is_same<T, half>::value) {
		return (half)std::exp((float)a);
	} else {
		return std::exp(a);
	}
})

DEFINE_UNARY_OP(tensor_log, [](T a) -> T {
	if constexpr(std::is_same<T, half>::value) {
		return (half)std::log((float)a);
	} else {
		return std::log(a);
	}
})

}