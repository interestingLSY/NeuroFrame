#pragma once

#include <vector>

#include "src/tensor/tensor.h"

namespace NeuroFrame {

// OpContext: The context when running an operator.
// It provides a function, `save_for_backward`, to save tensors for backward.
class OpContext {
	std::vector<Tensor> saved_tensors;

public:
	OpContext();
	void save_for_backward(const Tensor &tensor);
	std::vector<Tensor> get_saved_tensors() const;
};

}
