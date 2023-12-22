#include "optim.h"

#include <pybind11/stl.h>
using namespace pybind11::literals;	// For "_a" suffix

#include "src/optim/optim.h"
#include "src/optim/optim_state.h"
#include "src/optim/adam.h"
#include "src/optim/sgd.h"

using namespace NeuroFrame;
void init_optim(pybind11::module& m) {
	auto optim_m = m.def_submodule("optim", "NeuroFrame computational graph");

	pybind11::class_<SGDOptimizer>(optim_m, "SGD")
		.def("__init__", [](SGDOptimizer &self) {
			new (&self) SGDOptimizer();
		})
		.def("add_focus", &SGDOptimizer::add_focus, "tensor"_a)
		.def("remove_focus", &SGDOptimizer::remove_focus, "tensor"_a)
		.def("step", &SGDOptimizer::step, "learning_rate"_a);
	
	pybind11::class_<AdamOptimizer>(optim_m, "Adam")
		.def("__init__", [](AdamOptimizer &self, double beta1, double beta2, double eps) {
			new (&self) AdamOptimizer(beta1, beta2, eps);
		}, "beta1"_a = 0.9, "beta2"_a = 0.999, "eps"_a = 1e-8)
		.def("add_focus", &AdamOptimizer::add_focus, "tensor"_a)
		.def("remove_focus", &AdamOptimizer::remove_focus, "tensor"_a)
		.def("step", &AdamOptimizer::step, "learning_rate"_a);
}
