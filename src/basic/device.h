#pragma once

#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace NeuroFrame {

enum class device_type_t {
	CPU,
	CUDA
};

// Device: The abstraction of one device that is able to run kernels
class Device {
public:
	device_type_t type;

	// CUDA only
	int device_index;
	cudaStream_t stream;

	Device(device_type_t type, int device_index = 0);
	void switch_to() const;
	std::string to_string() const;
	std::string repr() const;

	inline static Device cpu() {
		return Device(device_type_t::CPU);
	}
	inline static Device cuda(int device_index = 0) {
		return Device(device_type_t::CUDA, device_index);
	}

	inline bool operator==(const Device& other) const {
		return type == other.type && device_index == other.device_index;
	}

	inline bool operator!=(const Device& other) const {
		return !(*this == other);
	}

	std::string get_hardware_name() const;
	
	static Device default_device;
	static std::vector<Device> get_available_devices();
	static Device get_default_device();
	static void set_default_device(const Device &device);
};

}
