#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "log.h"

namespace NeuroFrame {

enum class dtype_t {
	FLOAT16,
	FLOAT32,
	FLOAT64,
	INT8,
	INT16,
	INT32,
	INT64
};

inline bool is_float_family(dtype_t dtype) {
	return dtype == dtype_t::FLOAT16 || dtype == dtype_t::FLOAT32 || dtype == dtype_t::FLOAT64;
}

inline bool is_int_family(dtype_t dtype) {
	return dtype == dtype_t::INT8 || dtype == dtype_t::INT16 || dtype == dtype_t::INT32 || dtype == dtype_t::INT64;
}

std::string dtype2string(dtype_t dtype);

inline size_t get_dtype_size(dtype_t dtype) {
	switch (dtype) {
		case dtype_t::FLOAT16:
			return 2;
		case dtype_t::FLOAT32:
			return 4;
		case dtype_t::FLOAT64:
			return 8;
		case dtype_t::INT8:
			return 1;
		case dtype_t::INT16:
			return 2;
		case dtype_t::INT32:
			return 4;
		case dtype_t::INT64:
			return 8;
		default:
			LOG_FATAL("Unknown dtype");
	}
}

constexpr size_t MAX_DTYPE_SIZE = 8;

// Scalar: The abstraction of a scalar, which can be in different dtypes
class Scalar {
	dtype_t dtype;
	union x_t {
		double f;
		int64_t i;
	} x;

public:
	// Create a scalar from a pointer and a dtype
	Scalar(void* ptr, dtype_t dtype);

	// Create a scalar from a half
	Scalar(half f);
	// Create a scalar from a float
	Scalar(float f);
	// Create a scalar from a double
	Scalar(double f);

	// Create a scalar from an int8
	Scalar(int8_t i);
	// Create a scalar from an int16
	Scalar(int16_t i);
	// Create a scalar from an int32
	Scalar(int32_t i);
	// Create a scalar from an int64
	Scalar(int64_t i);

	// Create a scalar from something from the float family and the given dtype
	Scalar(double f, dtype_t dtype);

	// Create a scalar from something from the int family and the given dtype
	Scalar(int64_t i, dtype_t dtype);

	// Check if the scalar is in the float family (fp64, fp32, fp16)
	inline bool is_float_family() const {
		return NeuroFrame::is_float_family(dtype);
	}

	// Check if the scalar is in the int family (int64, int32, int16, int8)
	inline bool is_int_family() const {
		return NeuroFrame::is_int_family(dtype);
	}

	// Retrieve the value as double
	double as_double() const;

	// Retrieve the value as int64_t
	int64_t as_int64() const;

	// Save the value to a pointer, which has dtype = target_dtype
	void save_to(void* ptr, dtype_t target_dtype) const;

	std::string to_string() const;

	bool operator==(const Scalar &other) const;
};

}