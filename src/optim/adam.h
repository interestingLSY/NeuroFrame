#pragma once

#include "src/tensor/tensor.h"
#include "optim.h"
#include "optim_state.h"

namespace NeuroFrame {

class AdamOptimState : public OptimStateBase {
public:
	Tensor momentum;
	Tensor geo_mean;
	int cur_timestep;
	
	AdamOptimState(const Tensor &tensor);
	~AdamOptimState();
};

class AdamOptimizer : public Optimizer {
public:
	double beta1, beta2, eps;
	
	AdamOptimizer(double beta1, double beta2, double eps);
	~AdamOptimizer();

	void add_focus(const Tensor &tensor);
	void remove_focus(const Tensor &tensor);

	void step(double learning_rate);
};

}