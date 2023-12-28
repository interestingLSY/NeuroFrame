#include "misc.h"

#include <pybind11/stl.h>
using namespace pybind11::literals;	// For "_a" suffix

#include "src/basic/mem.h"
#include "src/tensor/tensor.h"
#include "src/utils/utils.h"
#include "src/utils/cuda_utils.h"

using namespace NeuroFrame;
void init_misc(pybind11::module& m) {
	auto ops_m = m.def_submodule("misc", "NeuroFrame misc operations");
	
	// Copy a torch tensor to a NeuroFrame tensor
	ops_m.def("copy_cpu_torch_tensor_to_gpu_nf_tensor", [](int64_t data_ptr, std::vector<int64_t> shape, dtype_t dtype) -> Tensor {
		int64_t numel = get_product_over_vector(shape);
		Tensor result = Tensor(shape, dtype, Device::cuda());
		CUDA_CHECK(cudaMemcpyAsync(
			result.data_ptr(),
			(void*)data_ptr,
			numel * get_dtype_size(dtype),
			cudaMemcpyHostToDevice
		));
		return result;
	}, "data_ptr"_a, "shape"_a, "dtype"_a);
}