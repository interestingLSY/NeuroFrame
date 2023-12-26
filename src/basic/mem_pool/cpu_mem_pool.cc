#include "cpu_mem_pool.h"

namespace NeuroFrame {

CPUMemPool::CPUMemPool(const Device device) : AbstractMemPool(device) {
}

CPUMemPool::~CPUMemPool() {
}

void* CPUMemPool::allocate(size_t length) {
	void* res = malloc(length);
	if (res == nullptr) {
		throw std::bad_alloc();
	}
	return res;
}

void CPUMemPool::free(void* ptr, size_t length) {
	::free(ptr);
}

}
