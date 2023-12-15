#pragma once

#include <cuda_fp16.h>

namespace NeuroFrame::Backend {

#define HALF_MIN ((half)-65504.)
#define HALF_MAX ((half)65504.)
#define FLOAT_MIN ((float)-3.4028235e+38)
#define FLOAT_MAX ((float)3.4028235e+38)
#define DOUBLE_MIN ((double)-1.7976931348623157e+308)
#define DOUBLE_MAX ((double)1.7976931348623157e+308)

template<typename T>
__host__ __device__ constexpr T get_inv_remainder() {
	if constexpr (std::is_same<T, half>::value) {
		return 1e-4;
	} else if constexpr (std::is_same<T, float>::value) {
		return 1e-6;
	} else if constexpr (std::is_same<T, double>::value) {
		return 1e-10;
	} else {
		LOG_FATAL("Unsupported type");
	}
}

}
