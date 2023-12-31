#include <pybind11/pybind11.h>
using namespace pybind11::literals;	// For "_a" suffix

#include "basic.h"
#include "cgraph.h"
#include "misc.h"
#include "ops.h"
#include "optim.h"
#include "tensor.h"

PYBIND11_MODULE(neuroframe, m) {
	m.doc() = "NeuroFrame: A neural network framework written in C++";

	init_basic(m);
	init_cgraph(m);
	init_misc(m);
	init_ops(m);
	init_optim(m);
	init_tensor(m);
}
