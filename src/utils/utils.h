#pragma once

#include <cassert>
#include <ostream>
#include <sstream>
#include <vector>

namespace NeuroFrame {

// A function that allows us to print a vector to some stream
template<typename T>
inline std::ostream& operator<<(std::ostream &os, const std::vector<T> &vec) {
	os << "[";
	for (size_t i = 0; i < vec.size(); ++i) {
		os << vec[i];
		if (i != vec.size() - 1) {
			os << ", ";
		}
	}
	os << "]";
	return os;
}

// A function that prints a vector to a string
template<typename T>
inline std::string vec_to_string(const std::vector<T> &vec) {
	std::stringstream ss;
	ss << vec;
	return ss.str();
}

// Calculate the coordinate of a element based on the shape & index
// For example, if shape = [2, 3, 4], index = 7, then the coordinate is [0, 1, 3]
inline std::vector<int64_t> get_coord_from_index(const std::vector<int64_t> &shape, int64_t index) {
	std::vector<int64_t> coord(shape.size());
	for (int i = (int)shape.size() - 1; i >= 0; --i) {
		coord[i] = index % shape[i];
		index /= shape[i];
	}
	return coord;
}

// Calculate the index of a element based on the shape & coordinate
// For example, if shape = [2, 3, 4], coord = [0, 1, 3], then the index is 7
inline int64_t get_index_from_coord(const std::vector<int64_t> &shape, const std::vector<int64_t> &coord) {
	int64_t index = 0;
	for (size_t i = 0; i < shape.size(); ++i) {
		index = index * shape[i] + coord[i];
	}
	return index;
}

// Calculate the stride of a tensor based on its shape
inline std::vector<int64_t> get_stride_from_shape(const std::vector<int64_t> &shape) {
	if (shape.size() == 0) {
		return {};
	}
	std::vector<int64_t> stride(shape.size());
	stride[shape.size() - 1] = 1;
	for (int i = (int)shape.size() - 2; i >= 0; --i) {
		stride[i] = stride[i + 1] * shape[i + 1];
	}
	return stride;
}

// Calculates the product of elements in a vector
inline int64_t get_product_over_vector(const std::vector<int64_t> &vec, int64_t start = 0, int64_t end = -1) {
	if (end == -1) {
		end = vec.size();
	}
	int64_t prod = 1;
	for (int64_t i = start; i < end; ++i) {
		prod *= vec[i];
	}
	return prod;
}

// Thresholds for floating point comparison
// When two floating point numbers satisfy fabs(a-b) <= abs_tol + rel_tol*max(fabs(a), fabs(b))
// they are considered equal
#define HALF_ABS_THRES ((half)2e-2)
#define HALF_REL_THRES ((half)1e-1)
#define FLOAT_ABS_THRES ((float)2e-4)
#define FLOAT_REL_THRES ((float)1e-2)
#define DOUBLE_ABS_THRES ((double)1e-6)
#define DOUBLE_REL_THRES ((double)1e-4)

/*
assert_whenever: assertion which ignore whether NDEBUG is set

In C++, assert() is evaluated only when NDEBUG is not set. This is
inconvenient when we want to check the assertion even in release mode.
This macro is a workaround for this problem.
*/

extern "C" {
// Copied from assert.h
extern void __assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
     __THROW __attribute__ ((__noreturn__));

#define __ASSERT_FUNCTION	__extension__ __PRETTY_FUNCTION__
#  define assert_whenever(expr)							\
     (static_cast <bool> (expr)						\
      ? void (0)							\
      : __assert_fail (#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))
}

}