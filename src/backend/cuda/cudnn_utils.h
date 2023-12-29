#pragma once

#include <cudnn.h>

#include "src/basic/scalar.h"
#include "src/basic/log.h"

namespace NeuroFrame::Backend::CUDA {

extern cudnnHandle_t cudnn_handle;

#define CUDNN_CHECK(call) \
{ \
	cudnnStatus_t status = call; \
	if (status != CUDNN_STATUS_SUCCESS) { \
		LOG_FATAL("CUDNN error: %s", cudnnGetErrorString(status)); \
	} \
}

inline cudnnDataType_t get_cudnn_data_type(dtype_t dtype) {
	switch (dtype) {
		case dtype_t::FLOAT16:
			return CUDNN_DATA_HALF;
		case dtype_t::FLOAT32:
			return CUDNN_DATA_FLOAT;
		case dtype_t::FLOAT64:
			return CUDNN_DATA_DOUBLE;
		default:
			LOG_FATAL("Unsupported dtype");
	}
}

std::pair<void*, void*> get_alpha_beta_ptrs(const dtype_t dtype);

}