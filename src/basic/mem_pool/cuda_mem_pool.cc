#include "cuda_mem_pool.h"

#include "src/utils/cuda_utils.h"
#include "src/basic/log.h"

namespace NeuroFrame {

CUDAMemPool::CUDAMemPool(const Device device) : AbstractMemPool(device) {
}

CUDAMemPool::~CUDAMemPool() {
	this->free_entire_free_list();
}

void CUDAMemPool::free_entire_free_list() {
	LOG_DEBUG("CUDAMemPool: Freeing entire free list for device %s", this->device.to_string().c_str());
	this->device.switch_to();
	for (auto& pair : this->free_list) {
		for (auto ptr : pair.second) {
			CUDA_CHECK(cudaFree(ptr));
		}
	}
	this->free_list.clear();
}

void* CUDAMemPool::allocate(size_t length) {
	if (length == 0) {
		LOG_FATAL("CUDAMemPool: Cannot allocate 0 bytes.");
	}
	if (this->free_list.count(length) != 0) {
		std::vector<void*> free_list = this->free_list[length];
		if (!free_list.empty()) {
			// If there is a free block of the required size, use it
			void* res = this->free_list[length].back();
			this->free_list[length].pop_back();
			return res;
		}
	}
	// Otherwise, allocate a new block
	this->device.switch_to();
	void* res;
	cudaError_t err = cudaMalloc(&res, length);
	if (err == cudaErrorMemoryAllocation) {
		this->free_entire_free_list();
		CUDA_CHECK(cudaMalloc(&res, length));
	} else {
		CUDA_CHECK(err);
	}
	return res;
	
}

void CUDAMemPool::free(void* ptr, size_t length) {
	this->free_list[length].push_back(ptr);
}

}
