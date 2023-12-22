#pragma once

#include <inttypes.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "src/basic/device.h"
#include "src/basic/mem.h"
#include "src/basic/scalar.h"

namespace NeuroFrame::CGraph {
	// Declare CGraphNode here to avoid circular dependency
	class CGraphNode;
}

namespace NeuroFrame {

// get_product_over_vector: Return the product of all elements in a vector
// If the vector is empty, return 1 (this follows the rule that "scalar tensor"
// (i.e. tensors with zero dimensions) has a numel of 1)
template<typename T>
inline T get_product_over_vector(const std::vector<T> &data) {
	T ret = 1;
	for (auto i : data) {
		ret *= i;
	}
	return ret;
}

class Tensor {
private:
	Tensor(const MemFrag &frag, const Device &dev, const dtype_t &dtype, const int64_t &first_elem_offset, const std::vector<int64_t> &shape);

	// Get the offset (in elements) of a given position
	int64_t get_elem_offset(const std::vector<int64_t> &pos) const;

public:
	MemFrag mem_frag;	// The fragment that stores the underlying data
	Device device;	// The device that the tensor is on
	dtype_t dtype;	// The data type of the tensor
	
	Tensor(const Tensor &other) = default;
	// Assign a scalar to a tensor (only applicable to scalar tensors)
	Tensor& operator=(const Scalar &other);

	// The following three fields (`first_elem_offset`, `shape`, and `stride`)
	// controls how the tensor is viewed.
	//
	// `first_elem_offset` is the offset of the first element of the tensor in
	// the underlying memory fragment. `shape` is the shape of the tensor.
	// `stride[i]` refers to the distance (in elements, not bytes) between two
	// adjacent elements in the i-th dimension.
	//
	// Different from PyTorch, stride[i] always equals to the product of
	// shape[i+1], shape[i+2], ..., shape[n-1], where n is the number of
	// dimensions of the tensor. In other words, the tensor is always
	// "continuous" in memory. I think this is a good design because it makes
	// the implementation of kernels much easier.
	int64_t first_elem_offset;
	std::vector<int64_t> shape;
	std::vector<int64_t> stride;

	// The corresponding CGraphNode in the compute graph
	std::shared_ptr<CGraph::CGraphNode> cgraph_node;

	// Basic operations
	// Return the number of elements in this tensor
	int64_t numel() const;
	// Return the start address
	void* data_ptr() const;
	// Return the dimension (i.e. the length of `shape`)
	int64_t dim() const;
	// Reshape
	Tensor reshape(const std::vector<int64_t> &new_shape) const;
	// Copy (deep copy)
	Tensor copy() const;
	// Replace the underlying memory fragment with a new one
	void replace_mem_frag(const MemFrag &new_frag);

	// Get the address of one element
	void* get_elem_addr(const std::vector<int64_t> &pos) const;
	// Get the content (in NeuroFrame::Tensor with shape == []) of one element
	Tensor get_elem(const std::vector<int64_t> &pos) const;
	// Return the element as a scalar. Only applicable to scalar tensors (tensors with shape == [])
	Scalar as_scalar() const;
	
	// Migration between devices
	Tensor to(Device target) const;
	// The same as `to(Device::cpu())`
	Tensor cpu() const;
	// The same as `to(Device::cuda(device_index))`
	Tensor cuda(int device_index = 0) const;
	
	std::string to_string(int64_t max_display_per_dim = 16, bool in_compat_style = false) const;
	std::string repr() const;
	void print(int64_t max_display_per_dim = 16, bool in_compat_style = false) const;

	// The following functions generate a tensor with the given shape, dtype and device
	// Do not fill
	Tensor(const std::vector<int64_t> &shape, dtype_t dtype, Device device);
	// Fill with zeros
	static Tensor zeros(const std::vector<int64_t> &shape, dtype_t dtype, Device device);
	static Tensor zeros_like(const Tensor &other);
	// Fill with the given number
	static Tensor fill(Scalar x, const std::vector<int64_t> &shape, dtype_t dtype, Device device);
	static Tensor fill_like(Scalar x, const Tensor &other);
	// Fill with uniform distribution between `low` and `high`
	static Tensor randu(const std::vector<int64_t> &shape, dtype_t dtype, Device device, Scalar low, Scalar high);
	// Fill with uniform distribution between -1 and +1
	static Tensor randu(const std::vector<int64_t> &shape, dtype_t dtype, Device device);
	// Fill with uniform distribution of integers between `low` and `high` (inclusive)
	static Tensor randint(const std::vector<int64_t> &shape, dtype_t dtype, Device device, Scalar low, Scalar high);
	// Create a tensor from a vector
	static Tensor from_vector(const std::vector<Scalar> &data, const std::vector<int64_t> &shape, dtype_t dtype, Device device);

	bool operator==(const Tensor &other) const;
	bool operator!=(const Tensor &other) const;

	Tensor operator+(const Tensor &other) const;
	Tensor operator-(const Tensor &other) const;
	Tensor operator*(const Tensor &other) const;
	Tensor operator/(const Tensor &other) const;
	Tensor operator-() const; 
};

std::ostream& operator<<(std::ostream &os, const Tensor &tensor);

}