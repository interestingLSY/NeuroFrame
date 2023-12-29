#include "pool.h"

#include <cmath>

#include "utils.h"

namespace NeuroFrame::Backend::CPU {

Tensor pool_forward(const Tensor &input, int pool_size, int stride, int padding) {
	LOG_FATAL("Not implemented");
}

Tensor pool_backward(const Tensor &output_grad, const Tensor &input, const Tensor &output, int pool_size, int stride, int padding) {
	LOG_FATAL("Not implemented");
}


}

