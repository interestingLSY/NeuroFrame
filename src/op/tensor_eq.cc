#include "tensor_eq.h"

#include "src/basic/log.h"

#include "src/backend/cpu/tensor_eq.h"
#include "src/backend/cuda/tensor_eq.h"


namespace NeuroFrame {

bool tensor_eq(const Tensor &input1, const Tensor &input2) {
	if (input1.device != input2.device) {
		LOG_FATAL("Cannot compare tensors on different devices");
	}
	if (input1.dtype != input2.dtype) {
		LOG_FATAL("Cannot compare tensors of different data types");
	}
	if (input1.shape != input2.shape) {
		return false;
	}
	bool result = DISPATCH_TO_BACKEND(
		input1.device.type,
		tensor_eq(input1, input2)
	);
	return result;
}

}
