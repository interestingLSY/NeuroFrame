#pragma once

#include "src/basic/device.h"

namespace NeuroFrame {

class AbstractMemPool {
protected:
	Device device;

public:
	AbstractMemPool(const Device device);

	virtual void* allocate(size_t length) = 0;
	virtual void free(void* ptr, size_t length) = 0;
};

}
