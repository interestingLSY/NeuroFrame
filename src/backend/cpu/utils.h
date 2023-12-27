#pragma once

#include <vector>

#include <cuda_fp16.h>

#include "src/backend/utils.h"

namespace NeuroFrame::Backend::CPU {
	
#define DISPATCH_ON_DTYPE_CPU_BACKEND(dtype, call) \
	[&]() { \
		switch (dtype) { \
			case dtype_t::FLOAT16: \
				{typedef half T; return call;} \
			case dtype_t::FLOAT32: \
				{typedef float T; return call;} \
			case dtype_t::FLOAT64: \
				{typedef double T; return call;} \
			default: \
				LOG_FATAL("Unknown dtype."); \
		} \
	}()

template<typename T>
static constexpr T get_min() {
	if constexpr (std::is_same<T, half>::value) {
		return HALF_MIN;
	} else if constexpr (std::is_same<T, float>::value) {
		return FLOAT_MIN;
	} else if constexpr (std::is_same<T, double>::value) {
		return DOUBLE_MIN;
	} else {
		return 0;
	}
}

template<typename T>
static constexpr T get_max() {
	if constexpr (std::is_same<T, half>::value) {
		return HALF_MAX;
	} else if constexpr (std::is_same<T, float>::value) {
		return FLOAT_MAX;
	} else if constexpr (std::is_same<T, double>::value) {
		return DOUBLE_MAX;
	} else {
		return 0;
	}
}

// Some stuff for indexing into an 1-D array
#define INDEX_2D(dim1, dim2, index1, index2) \
    (((int64_t)index1) * (dim2) + (index2))
#define INDEX_3D(dim1, dim2, dim3, index1, index2, index3) \
    (((int64_t)index1) * (dim2) * (dim3) + ((int64_t)index2) * (dim3) + (index3))
#define INDEX_4D(dim1, dim2, dim3, dim4, index1, index2, index3, index4) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) + ((int64_t)index2) * (dim3) * (dim4) + ((int64_t)index3) * (dim4) + (index4))
#define INDEX_5D(dim1, dim2, dim3, dim4, dim5, index1, index2, index3, index4, index5) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) * (dim5) + ((int64_t)index2) * (dim3) * (dim4) * (dim5) + ((int64_t)index3) * (dim4) * (dim5) + (index4) * (dim5) + (index5))

}