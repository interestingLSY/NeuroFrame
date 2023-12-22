#pragma once

#include "src/tensor/tensor.h"
#include "src/cgraph/cgraph_node.h"

namespace NeuroFrame {

// Optimizer - The abstract class for all optimizers (SGD, Adam, etc.)
class Optimizer {
protected:
	std::vector<Tensor> focused_nodes;

	// add_to_focus_list: Add a tensor to the optimizer's focus list
	void add_to_focus_list(const Tensor &tensor);

	// remove_from_focus_list: Remove a tensor from the optimizer's focus list
	void remove_from_focus_list(const Tensor &tensor);

public:
	Optimizer();
	virtual ~Optimizer();

	virtual void add_focus(const Tensor &tensor) = 0;

	virtual void remove_focus(const Tensor &tensor) = 0;

	// step: Perform a step of optimization
	virtual void step(double learning_rate) = 0;
};

}
