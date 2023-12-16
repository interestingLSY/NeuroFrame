#include "device.h"

#include <cuda_runtime.h>

#include "log.h"

namespace NeuroFrame {

Device::Device(device_type_t type, int device_index):
	type(type),
	device_index(device_index)
{
	if (type != device_type_t::CUDA && device_index != 0) {
		LOG_FATAL("Only device_index 0 is supported for CPU");
	}
	// Create the CUDA stream if necessary
	if (type == device_type_t::CUDA) {
		this->switch_to();
		cudaError_t err = cudaStreamCreate(&stream);
		if (err != cudaSuccess) {
			print_cuda_error();
			LOG_FATAL("Failed to create CUDA stream");
		}
	}
}

void Device::switch_to() const {
	if (type == device_type_t::CUDA) {
		cudaError_t err = cudaSetDevice(device_index);
		if (err != cudaSuccess) {
			print_cuda_error();
			LOG_FATAL("Failed to switch to device cuda:%d", device_index);
		}
	}
}

std::string Device::to_string() const {
	if (type == device_type_t::CPU) {
		return "cpu";
	} else if (type == device_type_t::CUDA) {
		return "cuda:" + std::to_string(device_index);
	} else {
		LOG_FATAL("Unknown device type");
	}
}

std::string Device::repr() const {
	return "<Device " + this->to_string() + ">";
}

Device Device::default_device = Device::cpu();

std::vector<Device> Device::get_available_devices() {
	std::vector<Device> devices;
	devices.push_back(Device::cpu());
	int device_count;
	cudaError_t err = cudaGetDeviceCount(&device_count);
	if (err != cudaSuccess) {
		print_cuda_error();
		LOG_FATAL("Failed to get CUDA device count");
	}
	for (int i = 0; i < device_count; ++i) {
		devices.push_back(Device::cuda(i));
	}
	return devices;
}

Device Device::get_default_device() {
	return default_device;
}

void Device::set_default_device(const Device &device) {
	default_device = device;
}

}