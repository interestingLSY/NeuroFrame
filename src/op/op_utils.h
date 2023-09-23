#pragma once

#include <vector>

#include "src/basic/device.h"
#include "src/tensor/tensor.h"
#include "op.h"
#include "op_context.h"

namespace NeuroFrame {

// Check if all tensors are on the same device.
bool is_on_same_device(const std::vector<Tensor> &tensors);

// Check if all tensors have the same dtype.
bool have_same_dtype(const std::vector<Tensor> &tensors);

// Check if all tensors are on the same device and have the same dtype.
// If either of them is not satisfied, panic.
// Used in forward and backward.
void do_basic_checkings_in_forward_and_backward(const std::vector<Tensor> &input, const OpContext &ctx);

#define DISPATCH_TO_BACKEND(device_type, call) \
	[&]() { \
		switch (device_type) { \
			case device_type_t::CPU: \
				return NeuroFrame::Backend::CPU:: call; \
			case device_type_t::CUDA: \
				return NeuroFrame::Backend::CUDA:: call; \
			default: \
				LOG_FATAL("Unknown device."); \
		} \
	}()
}
