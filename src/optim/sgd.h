#pragma once

#include "optim.h"
#include "optim_state.h"

namespace NeuroFrame {

class SGDOptimState : public OptimStateBase {
public:
	Tensor momentum;

	SGDOptimState(const Tensor &weight, bool have_momentum);

	~SGDOptimState();
};

class SGDOptimizer : public Optimizer {
private:
	double momentum;
	double weight_decay;
	
public:
	SGDOptimizer(double momentum, double weight_decay);
	~SGDOptimizer();

	void add_focus(const Tensor &tensor);
	void remove_focus(const Tensor &tensor);

	void step(double learning_rate);
};

}