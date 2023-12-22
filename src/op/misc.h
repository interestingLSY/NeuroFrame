#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

int64_t get_correct_sample_count(const Tensor &output, const Tensor &ground_truth);

}