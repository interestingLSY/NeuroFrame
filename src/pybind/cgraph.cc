#include "cgraph.h"

#include <pybind11/stl.h>
using namespace pybind11::literals;	// For "_a" suffix

#include "src/cgraph/cgraph.h"

using namespace NeuroFrame;
void init_cgraph(pybind11::module& m) {
	auto ops_m = m.def_submodule("cgraph", "NeuroFrame computational graph");

	ops_m.def("perform_backward", &CGraph::perform_backward, "src"_a, "src_grad"_a, "log_down_all_grads"_a = false, "Perform backward pass on the computational graph");

	// ops_m.def("get_topology_sort", &CGraph::get_topology_sort, "src"_a, "Get the topology sort of the computational graph");

	ops_m.def("get_computed_grad", &CGraph::get_computed_grad, "src"_a, "Get the computed gradient of the tensor");
	
	ops_m.def("clear_graph", &CGraph::clear_graph, "Clear the computational graph");
}