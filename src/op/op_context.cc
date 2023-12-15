#include "op_context.h"

namespace NeuroFrame {

OpContext::OpContext() {
	saved_args = nullptr;
}

OpContext::~OpContext() {
	if (saved_args != nullptr) {
		delete[] (char*)saved_args;
	}
}

void OpContext::save_args(void* args_ptr, size_t args_size) {
	if (saved_args != nullptr) {
		LOG_ERROR("save_args can only be called once");
	}
	saved_args = new char[args_size];
	::memcpy(saved_args, args_ptr, args_size);
}

void* OpContext::get_saved_args() const {
	return saved_args;
}

void OpContext::save_for_backward(const Tensor &tensor) {
	saved_tensors.emplace_back(tensor);
}

std::vector<Tensor> OpContext::get_saved_tensors() const {
	return saved_tensors;
}

Tensor OpContext::get_saved_tensor(int64_t index) const {
	return saved_tensors[index];
}

}
