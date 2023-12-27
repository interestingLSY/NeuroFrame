#include "cuda_mem_pool.h"

#include "src/utils/cuda_utils.h"
#include "src/basic/log.h"

namespace NeuroFrame {

CUDAMemPool::CUDAMemPool(const Device device) : AbstractMemPool(device) {
	this->device.switch_to();
	cudaDeviceGetDefaultMemPool(&mempool, device.device_index);

	uint64_t threshold = UINT64_MAX;
	cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

CUDAMemPool::~CUDAMemPool() {
}

void* CUDAMemPool::allocate(size_t length) {
	void* res;
	CUDA_CHECK(cudaMallocAsync(&res, length, cudaStreamDefault));
	return res;
	
}

void CUDAMemPool::free(void* ptr, size_t length) {
	CUDA_CHECK(cudaFreeAsync(ptr, cudaStreamDefault));
}

}
