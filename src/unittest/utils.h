#pragma once

#include <iostream>
#include <cmath>

#include "src/tensor/tensor.h"
#include "src/utils/utils.h"

namespace NeuroFrame {

// Check whether two floats are equal
// When two floating point numbers satisfy fabs(a-b) <= abs_tol + rel_tol*max(fabs(a), fabs(b))
// they are considered equal
inline bool is_float_equal(double a, double b, dtype_t dtype) {
	if (std::isnan(a) || std::isnan(b)) {
		LOG_WARN("NaN detected");
		return std::isnan(a) && std::isnan(b);
	}
	// std::isinf seems to have bug that it returns false when a == INFINITY
	auto my_isinf = [](double x) -> bool {
		return x == INFINITY || x == -INFINITY;
	};
	if (my_isinf(a) || std::isinf(b)) {
		LOG_WARN("Inf detected");
		if (!my_isinf(a) || !my_isinf(b)) {
			return false;
		}
		return (a > 0) == (b > 0);
	}
	double abs_tol, rel_tol;
	if (dtype == dtype_t::FLOAT16) {
		abs_tol = HALF_ABS_THRES;
		rel_tol = HALF_REL_THRES;
	} else if (dtype == dtype_t::FLOAT32) {
		abs_tol = FLOAT_ABS_THRES;
		rel_tol = FLOAT_REL_THRES;
	} else if (dtype == dtype_t::FLOAT64) {
		abs_tol = DOUBLE_ABS_THRES;
		rel_tol = DOUBLE_REL_THRES;
	} else {
		LOG_FATAL("Unknown dtype");
	}
	return std::abs(a-b) <= abs_tol + rel_tol*(std::max(std::abs(a), std::abs(b)) + 1e-6);
}

// Check whether two scalars are equal
inline bool is_scalar_equal(Scalar a, Scalar b) {
	if (a.dtype != b.dtype) {
		LOG_ERROR("The dtypes of two scalars are not the same: %s vs %s",
			dtype2string(a.dtype).c_str(), dtype2string(b.dtype).c_str());
		return false;
	}
	if (is_int_family(a.dtype)) {
		return a.as_int64() == b.as_int64();
	} else if (is_float_family(a.dtype)) {
		double a_v = a.as_double();
		double b_v = b.as_double();
		return is_float_equal(a_v, b_v, a.dtype);
	} else {
		LOG_FATAL("Unknown dtype");
	}
	return true;
}

// Check whether two tensors are equal
inline bool is_tensor_equal(Tensor a, Tensor b) {
	int64_t numel = a.numel();
	if (numel != b.numel()) {
		LOG_ERROR("The number of elements of two tensors are not the same: %ld vs %ld", 
			numel, b.numel());
		return false;
	}
	if (a.shape != b.shape) {
		LOG_ERROR("The shape of two tensors are not the same: %s vs %s",
			vec_to_string(a.shape).c_str(), vec_to_string(b.shape).c_str());
		return false;
	}
	a = a.to(Device::cpu());
	b = b.to(Device::cpu());
	int64_t mismatch_cnt = 0;
	for (int64_t index = 0; index < numel; ++index) {
		std::vector<int64_t> coord = get_coord_from_index(a.shape, index);
		Scalar a_elem = a.get_elem(coord).as_scalar();
		Scalar b_elem = b.get_elem(coord).as_scalar();
		if (!is_scalar_equal(a_elem, b_elem)) {
			if (mismatch_cnt < 10) {
				LOG_ERROR("The %s-th element of two tensors are not the same: %s vs %s",
					vec_to_string(coord).c_str(), a_elem.to_string().c_str(), b_elem.to_string().c_str());
			} else if (mismatch_cnt == 10) {
				LOG_ERROR("...");
			}
			mismatch_cnt += 1;
		}
	}
	if (mismatch_cnt != 0) {
		LOG_ERROR("(Total %ld mismatches)", mismatch_cnt);
		return false;
	}
	return true;
}

}