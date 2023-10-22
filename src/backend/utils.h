#pragma once

#include <cuda_fp16.h>

namespace NeuroFrame::Backend {

// Thresholds for floating point comparison
// When two floating point numbers satisfy fabs(a-b) <= abs_tol + rel_tol*max(fabs(a), fabs(b))
// they are considered equal
#define HALF_ABS_THRES ((half)2e-2)
#define HALF_REL_THRES ((half)1e-1)
#define FLOAT_ABS_THRES ((float)1e-4)
#define FLOAT_REL_THRES ((float)1e-2)
#define DOUBLE_ABS_THRES ((double)1e-6)
#define DOUBLE_REL_THRES ((double)1e-4)

#define HALF_MIN ((half)-65504.)
#define HALF_MAX ((half)65504.)
#define FLOAT_MIN ((float)-3.4028235e+38)
#define FLOAT_MAX ((float)3.4028235e+38)
#define DOUBLE_MIN ((double)-1.7976931348623157e+308)
#define DOUBLE_MAX ((double)1.7976931348623157e+308)

}
