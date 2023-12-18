#pragma once

#include <vector>

#include "src/tensor/tensor.h"

namespace NeuroFrame {

// OpContext: The context when running an operator.
// It provides a function, `save_for_backward`, to save tensors for backward.
class OpContext {
	std::vector<Tensor> saved_tensors;
	std::vector<char> saved_args;

public:
	OpContext();
	~OpContext();

	void save_args(void* args_ptr, size_t args_size);

	void* get_saved_args() const;

	void save_for_backward(const Tensor &tensor);
	
	std::vector<Tensor> get_saved_tensors() const;

	Tensor get_saved_tensor(int64_t index) const;
};

}
