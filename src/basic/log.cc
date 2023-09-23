#include "log.h"

#include <cuda_runtime.h>

namespace NeuroFrame {

void print_cuda_error() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
	}
}

}
