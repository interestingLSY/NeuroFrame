#pragma once

#include <cstddef>
#include <unordered_map>

#include "src/basic/device.h"
#include "scalar.h"

namespace NeuroFrame {

// MemFrag: a "block" of memory with reference counting
// A MemFrag can reside on either CPU or GPU, depending on the `device` field.
// When a mem frag is copied, the reference count is increased by 1.
// When a mem frag is destroyed, the reference count is decreased by 1.
// When the reference count reaches 0, the underlying memory is freed.
class MemFrag {
private:

public:
	NeuroFrame::Device device;	// The device that the memory fragment is on
	void* ptr;			// The start address
	size_t length;		// The length of the memory fragment
	size_t* refcnt;		// A pointer pointing to the reference count of the memory fragment.

	MemFrag(NeuroFrame::Device device, size_t length);
	~MemFrag();
	MemFrag(const MemFrag& other);
	MemFrag& operator=(const MemFrag& other);

	void copy_from(const MemFrag& other);
};

// memcpy - Copy memory between devices
void memcpy(void* dst_ptr, const Device &dst_dev, const void* src_ptr, const Device &src_dev, size_t length);

// memcpy_strided - Copy memory between devices with strides
// The strides are in bytes
void memcpy_strided(
	void* dst_ptr, const Device &dst_dev,
	const void* src_ptr, const Device &src_dev,
	size_t dst_stride, size_t src_stride,
	size_t block_len, size_t num_blocks
);

void memset(void* ptr, const Device &dev, size_t length, unsigned char value);

}
