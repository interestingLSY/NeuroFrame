#include "mem_pool.h"

#include <stdexcept>

#include "cpu_mem_pool.h"
#include "cuda_mem_pool.h"

namespace NeuroFrame {

static CPUMemPool* cpu_mem_pool = nullptr;
static std::vector<CUDAMemPool*> cuda_mem_pools = {};

void create_mem_pools() {
	// Create CPU mem pool
	cpu_mem_pool = new CPUMemPool(Device::cpu());
	// Create CUDA mem pools
	auto devices = Device::get_available_devices();
	for (const Device &device : devices) {
		if (device.type == device_type_t::CUDA) {
			cuda_mem_pools.push_back(new CUDAMemPool(device));
		}
	}
}

class _MemPoolCreator {
public:
	_MemPoolCreator() {
		create_mem_pools();
	}
} _mem_pool_creator;

AbstractMemPool* get_mem_pool(const Device &device) {
	if (device.type == device_type_t::CPU) {
		return cpu_mem_pool;
	} else if (device.type == device_type_t::CUDA) {
		return cuda_mem_pools[device.device_index];
	} else {
		throw std::runtime_error("Unknown device type");
	}
}

}