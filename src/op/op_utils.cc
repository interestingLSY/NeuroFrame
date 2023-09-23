#include "op_utils.h"

namespace NeuroFrame {

bool is_on_same_device(const std::vector<Tensor> &tensors) {
	if (tensors.empty()) {
		return true;
	}
	Device device_type = tensors[0].device;
	for (size_t i = 1; i < tensors.size(); i++) {
		if (tensors[i].device != device_type) {
			return false;
		}
	}
	return true;
}

bool have_same_dtype(const std::vector<Tensor> &tensors) {
	if (tensors.empty()) {
		return true;
	}
	dtype_t dtype = tensors[0].dtype;
	for (size_t i = 1; i < tensors.size(); i++) {
		if (tensors[i].dtype != dtype) {
			return false;
		}
	}
	return true;
}

void do_basic_checkings_in_forward_and_backward(const std::vector<Tensor> &input, const OpContext &ctx) {
	std::vector<Tensor> saved_tensors = ctx.get_saved_tensors();
	if (!is_on_same_device(input) ||
		!is_on_same_device(saved_tensors) ||
		(!input.empty() && !saved_tensors.empty() && input[0].device != saved_tensors[0].device)) {
		LOG_FATAL("All tensors must be on the same device.");
	}
	if (!have_same_dtype(input) ||
		!have_same_dtype(saved_tensors) ||
		(!input.empty() && !saved_tensors.empty() && input[0].dtype != saved_tensors[0].dtype)) {
		LOG_FATAL("All tensors must have the same dtype.");
	}
}

}