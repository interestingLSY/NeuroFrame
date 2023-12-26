#include "mem.h"

#include <string.h>

#include <memory>
#include <stdexcept>

#include "log.h"
#include "mem_pool/mem_pool.h"

namespace NeuroFrame {

MemFrag::MemFrag(NeuroFrame::Device device, size_t length):
	device(device),
	length(length)
{
	ptr = get_mem_pool(device)->allocate(length);
	refcnt = new size_t;
	*refcnt = 1;
}

MemFrag::~MemFrag() {
	*refcnt -= 1;
	if (*refcnt == 0) {
		delete refcnt;
		get_mem_pool(device)->free(ptr, length);
	}
}

MemFrag::MemFrag(const MemFrag& other):
	device(other.device),
	ptr(other.ptr),
	length(other.length),
	refcnt(other.refcnt)
{
	*refcnt += 1;
}

MemFrag& MemFrag::operator=(const MemFrag& other) {
	if (this == &other) {
		return *this;
	}
	*refcnt -= 1;
	if (*refcnt == 0) {
		delete refcnt;
		get_mem_pool(device)->free(ptr, length);
	}
	device = other.device;
	ptr = other.ptr;
	length = other.length;
	refcnt = other.refcnt;
	*refcnt += 1;
	return *this;
}

void MemFrag::copy_from(const MemFrag& other) {
	if (this->length != other.length) {
		LOG_FATAL("Cannot copy memory between memory fragments with different lengths: %zu and %zu", this->length, other.length);
	}
	memcpy(this->ptr, this->device, other.ptr, other.device, this->length);
}

void memcpy(void* dst_ptr, const Device &dst_dev, const void* src_ptr, const Device &src_dev, size_t length) {
	if (dst_dev.type == device_type_t::CPU && src_dev.type == device_type_t::CPU) {
		// CPU -> CPU
		::memcpy(dst_ptr, src_ptr, length);
	} else if (dst_dev.type == device_type_t::CPU && src_dev.type == device_type_t::CUDA) {
		// CUDA -> CPU
		src_dev.switch_to();
		cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, length, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			print_cuda_error();
			LOG_FATAL("Failed to copy memory from CUDA to CPU");
		}
	} else if (dst_dev.type == device_type_t::CUDA && src_dev.type == device_type_t::CPU) {
		// CPU -> CUDA
		dst_dev.switch_to();
		cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, length, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			print_cuda_error();
			LOG_FATAL("Failed to copy memory from CPU to CUDA");
		}
	} else if (dst_dev.type == device_type_t::CUDA && src_dev.type == device_type_t::CUDA) {
		// CUDA -> CUDA
		if (dst_dev.device_index != src_dev.device_index) {
			LOG_FATAL("Cannot copy memory between different CUDA devices");
		} else {
			// The two memory blocks reside on the same device
			cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, length, cudaMemcpyDeviceToDevice);
			if (err != cudaSuccess) {
				print_cuda_error();
				LOG_FATAL("Failed to copy memory from CUDA to CUDA");
			}
		}
	} else {
		LOG_FATAL("Unknown device type");
	}
}

void memcpy_strided(
	void* dst_ptr, const Device &dst_dev,
	const void* src_ptr, const Device &src_dev,
	size_t dst_stride, size_t src_stride,
	size_t block_len, size_t num_blocks
) {
	if (dst_dev.type == device_type_t::CPU && src_dev.type == device_type_t::CPU) {
		// CPU -> CPU
		for (size_t i = 0; i < num_blocks; i++) {
			::memcpy((char*)dst_ptr + i * dst_stride, (char*)src_ptr + i * src_stride, block_len);
		}
	} else if (dst_dev.type == device_type_t::CPU && src_dev.type == device_type_t::CUDA) {
		// CUDA -> CPU
		src_dev.switch_to();
		cudaError_t err = cudaMemcpy2D(
			dst_ptr, dst_stride,
			src_ptr, src_stride,
			block_len, num_blocks,
			cudaMemcpyDeviceToHost
		);
		if (err != cudaSuccess) {
			print_cuda_error();
			LOG_FATAL("Failed to copy memory from CUDA to CPU");
		}
	} else if (dst_dev.type == device_type_t::CUDA && src_dev.type == device_type_t::CPU) {
		// CPU -> CUDA
		dst_dev.switch_to();
		cudaError_t err = cudaMemcpy2D(
			dst_ptr, dst_stride,
			src_ptr, src_stride,
			block_len, num_blocks,
			cudaMemcpyHostToDevice
		);
		if (err != cudaSuccess) {
			print_cuda_error();
			LOG_FATAL("Failed to copy memory from CPU to CUDA");
		}
	} else if (dst_dev.type == device_type_t::CUDA && src_dev.type == device_type_t::CUDA) {
		// CUDA -> CUDA
		if (dst_dev.device_index != src_dev.device_index) {
			LOG_FATAL("Cannot copy memory between different CUDA devices");
		} else {
			// The two memory blocks reside on the same device
			cudaError_t err = cudaMemcpy2D(
				dst_ptr, dst_stride,
				src_ptr, src_stride,
				block_len, num_blocks,
				cudaMemcpyDeviceToDevice
			);
			if (err != cudaSuccess) {
				print_cuda_error();
				LOG_FATAL("Failed to copy memory from CUDA to CUDA");
			}
		}
	}
}

void memset(void* ptr, const Device &dev, size_t length, unsigned char value) {
	if (dev.type == device_type_t::CPU) {
		::memset(ptr, value, length);
	} else if (dev.type == device_type_t::CUDA) {
		dev.switch_to();
		cudaError_t err = cudaMemset(ptr, value, length);
		if (err != cudaSuccess) {
			print_cuda_error();
			LOG_FATAL("Failed to set memory on CUDA");
		}
	} else {
		LOG_FATAL("Unknown device type");
	}
}

}
