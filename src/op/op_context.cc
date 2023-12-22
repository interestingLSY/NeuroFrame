#include "op_context.h"

#include "src/basic/inference_mode.h"

namespace NeuroFrame {

OpContext::OpContext() {
	saved_args = {};
}

OpContext::~OpContext() {
}

void OpContext::save_args(void* args_ptr, size_t args_size) {
	if (!is_inference_mode()) {
		if (!saved_args.empty()) {
			LOG_FATAL("save_args can only be called once");
		}
		saved_args.resize(args_size);
		::memcpy(saved_args.data(), args_ptr, args_size);
	}
}

void* OpContext::get_saved_args() const {
	return (void*)saved_args.data();
}

void OpContext::save_for_backward(const Tensor &tensor) {
	if (!is_inference_mode()) {
		saved_tensors.emplace_back(tensor);
	}
}

std::vector<Tensor> OpContext::get_saved_tensors() const {
	return saved_tensors;
}

Tensor OpContext::get_saved_tensor(int64_t index) const {
	if (index < 0 || index >= (int64_t)saved_tensors.size()) {
		LOG_FATAL("Index out of range (%lu tensor saved, #%ld requested)", saved_tensors.size(), index);
	}
	return saved_tensors[index];
}

}
