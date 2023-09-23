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

Scalar::Scalar(double f, dtype_t dtype) {
	this->dtype = dtype;
	x.f = f;
}

double Scalar::as_double() const {
	if (is_float_family()) {
		return x.f;
	} else {
		LOG_FATAL("The scalar is not in the float family");
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
	} else {
		LOG_FATAL("Unknown dtype");
	}
}

std::string Scalar::to_string() const {
	if (is_float_family()) {
		return std::to_string(x.f);
	} else {
		LOG_FATAL("Unknown dtype");
	}
}

}