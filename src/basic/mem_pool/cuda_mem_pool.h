#pragma once

#include <unordered_map>

#include "src/basic/device.h"
#include "src/basic/mem_pool/abstract_mem_pool.h"

namespace NeuroFrame {

// CUDAMemPool - A simple GPU memory pool
// 
// This module provides interfaces for memory allocation & deallocation
// on GPU.
// 
// Its internal logic is really simple - Since, usually, during training
// or inferencing, we tend to allocate and free memory of some fixed
// sizes repeatedly, we can maintain a free list for each size, and
// allocate from the corresponding free list if possible. If the list
// is empty, we allocate a new block. If the allocation fails, we free
// all blocks in the free list and try again.
// 
// This pool really helps to reduce the overhead of memory allocation.
// We get a 2.5x speedup on MNIST-conv and a 2x speedup on MNIST-mlp
class CUDAMemPool: public AbstractMemPool {
private:
	std::unordered_map<size_t, std::vector<void*>> free_list;

	// Free up all blocks in the free list, and clear it
	void free_entire_free_list();

public:
	CUDAMemPool(const Device device);
	virtual ~CUDAMemPool();

	virtual void* allocate(size_t length);
	virtual void free(void* ptr, size_t length);
};

}
