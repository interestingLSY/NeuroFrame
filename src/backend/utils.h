#pragma once

#include <cuda_fp16.h>

namespace NeuroFrame::Backend {

// Thresholds for floating point comparison
// When two floating point numbers satisfy both absolute and relative
// thresholds, they are considered equal.
#define HALF_ABS_THRES ((half)1e-3)
#define HALF_REL_THRES ((half)1e-1)
#define FLOAT_ABS_THRES ((float)1e-4)
#define FLOAT_REL_THRES ((float)1e-2)
#define DOUBLE_ABS_THRES ((double)1e-6)
#define DOUBLE_REL_THRES ((double)1e-4)

}
