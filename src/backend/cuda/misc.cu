#include "misc.h"

#include <cstdint>
#include <omp.h>

#include "reduction.cuh"
#include "src/utils/cuda_utils.h"
#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

__device__ int32_t answer;

// grid: [batch_size]
// block: [num_classes]
template<typename T>
__global__ void get_correct_sample_count_kernel(
	const T* __restrict__ pred_output,			// [batch_size, num_classes]
	const int32_t* __restrict__ ground_truth,	// [batch_size]
	int64_t batch_size,
	int64_t num_classes
) {
	int64_t batch_id = blockIdx.x;
	int64_t class_id = threadIdx.x;

	T my_pred_output = pred_output[INDEX_2D(batch_size, num_classes, batch_id, class_id)];
	T group_max_pred_output = block_reduce_max_broadcast<T>(my_pred_output);

	int32_t my_max_pos = (my_pred_output == group_max_pred_output) ? class_id : -1;
	int32_t group_max_pos = block_reduce_max_broadcast<int32_t>(my_max_pos);

	if (group_max_pred_output == my_pred_output &&
		group_max_pos == class_id &&
		ground_truth[batch_id] == class_id) {
		atomicAdd(&answer, 1);
	}
}

int64_t get_correct_sample_count(const Tensor &pred_output, const Tensor &ground_truth) {
	int64_t batch_size = pred_output.shape[0];
	int64_t num_classes = pred_output.shape[1];

	int32_t* answer_dev_addr;
	cudaGetSymbolAddress((void**)&answer_dev_addr, answer);

	CUDA_CHECK(cudaMemset(answer_dev_addr, 0, sizeof(int32_t)));
	DISPATCH_ON_DTYPE_CUDA_BACKEND(
		pred_output.dtype,
		get_correct_sample_count_kernel<<<batch_size, num_classes>>>(
			(const T*) pred_output.data_ptr(),
			(const int32_t*) ground_truth.data_ptr(),
			batch_size,
			num_classes
		)
	);

	int32_t answer_h;
	CUDA_CHECK(cudaMemcpyFromSymbol(&answer_h, answer, sizeof(int32_t)));

	return answer_h;
}

template<typename T, bool HAVE_MOMENTUM, bool HAVE_WEIGHT_DECAY>
__global__ void sgd_grad_update_kernel(
	T* __restrict__ weight,
	const T* __restrict__ grad,
	T* __restrict__ momentum,
	T learning_rate,
	T momentum_factor,
	T weight_decay,
	int64_t num_elements
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x * gridDim.x) {
		T cur_grad = HAVE_WEIGHT_DECAY ? grad[i] + weight_decay * weight[i] : grad[i];
		if constexpr (HAVE_MOMENTUM) {
			cur_grad = momentum[i] = momentum_factor * momentum[i] + cur_grad;
		}
		weight[i] = weight[i] - learning_rate*cur_grad;
	}
}

void sgd_grad_update(Tensor &weight, const Tensor &grad, Tensor &momentum, double learning_rate, double momentum_factor, double weight_decay) {
	int64_t numel = weight.numel();
	int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int64_t grid_size = element_wise_kernel_get_num_grids(numel);
	#define DISPATCH(have_momentum, have_weight_decay) \
		DISPATCH_ON_DTYPE_CUDA_BACKEND(weight.dtype, \
			sgd_grad_update_kernel<T, have_momentum, have_weight_decay><<<grid_size, block_size>>>( \
				(T*) weight.data_ptr(), \
				(const T*) grad.data_ptr(), \
				(T*) momentum.data_ptr(), \
				(T) learning_rate, \
				(T) momentum_factor, \
				(T) weight_decay, \
				numel \
			));

	if (momentum_factor == 0) {
		if (weight_decay == 0) {
			DISPATCH(false, false);
		} else {
			DISPATCH(false, true);
		}
	} else {
		if (weight_decay == 0) {
			DISPATCH(true, false);
		} else {
			DISPATCH(true, true);
		}
	}
}

template<typename T>
__device__ __forceinline__ T my_sqrt(const T x) {
	if constexpr (std::is_same<T, half>::value) {
		return (T)sqrtf((float)x);
	} else if constexpr (std::is_same<T, float>::value) {
		return sqrtf(x);
	} else if constexpr (std::is_same<T, double>::value) {
		return sqrt(x);
	} else {
		assert(false);
	}
}

template<typename T>
__device__ __forceinline__ T my_pow(const T x, const T y) {
	if constexpr (std::is_same<T, half>::value) {
		return (T)powf((float)x, (float)y);
	} else if constexpr (std::is_same<T, float>::value) {
		return powf(x, y);
	} else if constexpr (std::is_same<T, double>::value) {
		return pow(x, y);
	} else {
		assert(false);
	}
}

template<typename T>
__global__ void adam_grad_update_kernel(
	T* __restrict__ weight,
	const T* __restrict__ grad,
	T* __restrict__ momentum,
	T* __restrict__ velocity,
	T learning_rate,
	T beta1,
	T beta2,
	T eps,
	int64_t cur_timestamp,
	int64_t num_elements
) {
	T cur_timestamp_T = (T)(long long)cur_timestamp;
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x * gridDim.x) {
		T cur_grad = grad[i];
		momentum[i] = beta1 * momentum[i] + ((T)1. - beta1) * cur_grad;
		velocity[i] = beta2 * velocity[i] + ((T)1. - beta2) * cur_grad * cur_grad;
		T adjusted_momentum = momentum[i] / ((T)1. - my_pow(beta1, cur_timestamp_T));
		T adjusted_velocity = velocity[i] / ((T)1. - my_pow(beta2, cur_timestamp_T));
		weight[i] = weight[i] - learning_rate * adjusted_momentum / (my_sqrt(adjusted_velocity) + eps);
	}
}

void adam_grad_update(Tensor &weight, const Tensor &grad, Tensor &momentum, Tensor &velocity, int cur_timestamp, double learning_rate, double beta1, double beta2, double eps) {
	int64_t numel = weight.numel();
	int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int64_t grid_size = element_wise_kernel_get_num_grids(numel);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(weight.dtype,
		adam_grad_update_kernel<T><<<grid_size, block_size>>>(
			(T*) weight.data_ptr(),
			(const T*) grad.data_ptr(),
			(T*) momentum.data_ptr(),
			(T*) velocity.data_ptr(),
			(T) learning_rate,
			(T) beta1,
			(T) beta2,
			(T) eps,
			cur_timestamp,
			numel
		)
	);
}

}
