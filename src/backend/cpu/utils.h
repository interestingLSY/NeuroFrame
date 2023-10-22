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

}