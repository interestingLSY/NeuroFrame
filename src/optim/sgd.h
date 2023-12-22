#pragma once

#include "optim.h"
#include "optim_state.h"

namespace NeuroFrame {

class SGDOptimState : public OptimStateBase {
public:
	SGDOptimState();
	~SGDOptimState();
};

class SGDOptimizer : public Optimizer {
public:
	SGDOptimizer();
	~SGDOptimizer();

	void add_focus(const Tensor &tensor);
	void remove_focus(const Tensor &tensor);

	void step(double learning_rate);
};

}