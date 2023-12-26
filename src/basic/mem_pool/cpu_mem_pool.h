#pragma once

#include "src/basic/device.h"
#include "src/basic/mem_pool/abstract_mem_pool.h"

namespace NeuroFrame {

class CPUMemPool: public AbstractMemPool {
public:
	CPUMemPool(const Device device);
	virtual ~CPUMemPool();

	virtual void* allocate(size_t length);
	virtual void free(void* ptr, size_t length);
};

}
