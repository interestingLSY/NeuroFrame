#pragma once

#include <vector>
#include <ostream>
#include <sstream>

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

// A function that calculates the coordinate of a element based on the shape & index
// For example, if shape = [2, 3, 4], index = 7, then the coordinate is [0, 1, 3]
inline std::vector<int64_t> get_coord_from_index(const std::vector<int64_t> &shape, int64_t index) {
	std::vector<int64_t> coord(shape.size());
	for (int i = shape.size() - 1; i >= 0; --i) {
		coord[i] = index % shape[i];
		index /= shape[i];
	}
	return coord;
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

}