#include "scalar.h"

namespace NeuroFrame {

std::string dtype2string(dtype_t dtype) {
	switch (dtype) {
		case dtype_t::FLOAT16:
			return "float16";
		case dtype_t::FLOAT32:
			return "float32";
		case dtype_t::FLOAT64:
			return "float64";
		case dtype_t::INT8:
			return "int8";
		case dtype_t::INT16:
			return "int16";
		case dtype_t::INT32:
			return "int32";
		case dtype_t::INT64:
			return "int64";
		default:
			LOG_FATAL("Unknown dtype");
	}
}

Scalar::Scalar(void* ptr, dtype_t dtype) {
	this->dtype = dtype;
	switch (dtype) {
		case dtype_t::FLOAT16:
			x.f = (double)*(half*)ptr;
			break;
		case dtype_t::FLOAT32:
			x.f = (double)*(float*)ptr;
			break;
		case dtype_t::FLOAT64:
			x.f = *(double*)ptr;
			break;
		case dtype_t::INT8:
			x.i = (int64_t)*(int8_t*)ptr;
			break;
		case dtype_t::INT16:
			x.i = (int64_t)*(int16_t*)ptr;
			break;
		case dtype_t::INT32:
			x.i = (int64_t)*(int32_t*)ptr;
			break;
		case dtype_t::INT64:
			x.i = *(int64_t*)ptr;
			break;
		default:
			LOG_FATAL("Unknown dtype");
	}
}
	
Scalar::Scalar(half f) {
	dtype = dtype_t::FLOAT16;
	x.f = (double)f;
}

Scalar::Scalar(float f) {
	dtype = dtype_t::FLOAT32;
	x.f = (double)f;
}

Scalar::Scalar(double f) {
	dtype = dtype_t::FLOAT64;
	x.f = f;
}

Scalar::Scalar(int8_t i) {
	dtype = dtype_t::INT8;
	x.i = (int64_t)i;
}

Scalar::Scalar(int16_t i) {
	dtype = dtype_t::INT16;
	x.i = (int64_t)i;
}

Scalar::Scalar(int32_t i) {
	dtype = dtype_t::INT32;
	x.i = (int64_t)i;
}

Scalar::Scalar(int64_t i) {
	dtype = dtype_t::INT64;
	x.i = i;
}

Scalar::Scalar(double f, dtype_t dtype) {
	this->dtype = dtype;
	x.f = f;
}

Scalar::Scalar(int64_t i, dtype_t dtype) {
	this->dtype = dtype;
	x.i = i;
}

double Scalar::as_double() const {
	if (is_float_family()) {
		return x.f;
	} else {
		return (double)x.i;
	}
}

int64_t Scalar::as_int64() const {
	if (is_int_family()) {
		return x.i;
	} else {
		return (int64_t)x.f;
	}
}

template<typename T>
T Scalar::to_c_dtype() const {
	if constexpr (std::is_same_v<T, half>) {
		return is_float_family() ? (half)x.f : (half)(float)x.i;
	} else if constexpr (std::is_same_v<T, float>) {
		return is_float_family() ? (float)x.f : (float)x.i;
	} else if constexpr (std::is_same_v<T, double>) {
		return is_float_family() ? x.f : (double)x.i;
	} else if constexpr (std::is_same_v<T, int8_t>) {
		return is_int_family() ? (int8_t)x.i : (int8_t)x.f;
	} else if constexpr (std::is_same_v<T, int16_t>) {
		return is_int_family() ? (int16_t)x.i : (int16_t)x.f;
	} else if constexpr (std::is_same_v<T, int32_t>) {
		return is_int_family() ? (int32_t)x.i : (int32_t)x.f;
	} else if constexpr (std::is_same_v<T, int64_t>) {
		return is_int_family() ? x.i : (int64_t)x.f;
	} else {
		LOG_FATAL("Unknown dtype");
	}
}

#define INSTANTIATE_TO_C_DTYPE(T) \
	template T Scalar::to_c_dtype<T>() const;
INSTANTIATE_TO_C_DTYPE(half)
INSTANTIATE_TO_C_DTYPE(float)
INSTANTIATE_TO_C_DTYPE(double)
INSTANTIATE_TO_C_DTYPE(int8_t)
INSTANTIATE_TO_C_DTYPE(int16_t)
INSTANTIATE_TO_C_DTYPE(int32_t)
INSTANTIATE_TO_C_DTYPE(int64_t)

Scalar Scalar::to_dtype(dtype_t target_dtype) const {
	if (is_float_family()) {
		if (NeuroFrame::is_float_family(target_dtype)) {
			return Scalar(x.f, target_dtype);
		} else if (NeuroFrame::is_int_family(target_dtype)) {
			if (target_dtype == dtype_t::INT8) {
				return Scalar((int8_t)x.f);
			} else if (target_dtype == dtype_t::INT16) {
				return Scalar((int16_t)x.f);
			} else if (target_dtype == dtype_t::INT32) {
				return Scalar((int32_t)x.f);
			} else if (target_dtype == dtype_t::INT64) {
				return Scalar((int64_t)x.f);
			} else {
				LOG_FATAL("Unknown dtype");
			}
		} else {
			LOG_FATAL("Unknown dtype");
		}
	} else if (is_int_family()) {
		if (NeuroFrame::is_int_family(target_dtype)) {
			return Scalar(x.i, target_dtype);
		} else if (NeuroFrame::is_float_family(target_dtype)) {
			if (target_dtype == dtype_t::FLOAT16) {
				return Scalar((half)(double)x.i);
			} else if (target_dtype == dtype_t::FLOAT32) {
				return Scalar((float)x.i);
			} else if (target_dtype == dtype_t::FLOAT64) {
				return Scalar((double)x.i);
			} else {
				LOG_FATAL("Unknown dtype");
			}
		} else {
			LOG_FATAL("Unknown dtype");
		}
	} else {
		LOG_FATAL("Unknown dtype");
	}
}

void Scalar::save_to(void* ptr, dtype_t target_dtype) const {
	if (is_float_family()) {
		if (!NeuroFrame::is_float_family(target_dtype)) {
			LOG_FATAL("Cannot save a scalar in the float family to a non-float-family dtype");
		}
		if (target_dtype == dtype_t::FLOAT16) {
			*(half*)ptr = (half)x.f;
		} else if (target_dtype == dtype_t::FLOAT32) {
			*(float*)ptr = (float)x.f;
		} else if (target_dtype == dtype_t::FLOAT64) {
			*(double*)ptr = x.f;
		} else {
			LOG_FATAL("Unknown dtype");
		}
	} else if (is_int_family()) {
		if (!NeuroFrame::is_int_family(target_dtype)) {
			LOG_FATAL("Cannot save a scalar in the int family to a non-int-family dtype");
		}
		if (target_dtype == dtype_t::INT8) {
			*(int8_t*)ptr = (int8_t)x.i;
		} else if (target_dtype == dtype_t::INT16) {
			*(int16_t*)ptr = (int16_t)x.i;
		} else if (target_dtype == dtype_t::INT32) {
			*(int32_t*)ptr = (int32_t)x.i;
		} else if (target_dtype == dtype_t::INT64) {
			*(int64_t*)ptr = x.i;
		} else {
			LOG_FATAL("Unknown dtype");
		}
	} else {
		LOG_FATAL("Unknown dtype");
	}
}

std::string Scalar::to_string() const {
	if (is_float_family()) {
		return std::to_string(x.f);
	} else if (is_int_family()) {
		return std::to_string(x.i);
	} else {
		LOG_FATAL("Unknown dtype");
	}
}

std::string Scalar::repr() const {
	return "<Scalar " + dtype2string(this->dtype) + " = " + this->to_string() + ">";
}

bool Scalar::operator==(const Scalar &other) const {
	if (dtype != other.dtype) {
		return false;
	}
	if (is_float_family()) {
		return x.f == other.x.f;
	} else if (is_int_family()) {
		return x.i == other.x.i;
	} else {
		LOG_FATAL("Unknown dtype");
	}
}

bool Scalar::operator!=(const Scalar &other) const {
	return !(*this == other);
}

}