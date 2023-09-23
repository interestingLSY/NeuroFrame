#pragma once

#include <vector>

#include <cuda_fp16.h>

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

}