#include "cudnn_utils.h"

namespace NeuroFrame::Backend::CUDA {

cudnnHandle_t cudnn_handle;

class CUDNNInitializer {
public:
	CUDNNInitializer() {
		CUDNN_CHECK(cudnnCreate(&cudnn_handle));
	}
	~CUDNNInitializer() {
		cudnnDestroy(cudnn_handle);
	}
} _cudnn_initializer;

std::pair<void*, void*> get_alpha_beta_ptrs(const dtype_t dtype) {
	static float float_one = 1.0, float_zero = 0.0;
	static double double_one = 1.0, double_zero = 0.0;
	void* alpha_ptr = dtype == dtype_t::FLOAT64 ? (void*)&double_one : (void*)&float_one;
	void* beta_ptr = dtype == dtype_t::FLOAT64 ? (void*)&double_zero : (void*)&float_zero;
	return {alpha_ptr, beta_ptr};
}

}