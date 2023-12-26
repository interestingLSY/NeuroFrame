#include "log.h"

#include <cuda_runtime.h>

namespace NeuroFrame {

namespace Logging {
	const int LOG_FATAL_BUF_SIZE = 128;
	char* log_fatal_buf = new char[LOG_FATAL_BUF_SIZE];
}

void print_cuda_error() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
	}
}

}
