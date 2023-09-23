#include "op_context.h"

namespace NeuroFrame {

OpContext::OpContext() {
}

void OpContext::save_for_backward(const Tensor &tensor) {
	saved_tensors.emplace_back(tensor);
}

std::vector<Tensor> OpContext::get_saved_tensors() const {
	return saved_tensors;
}

}
