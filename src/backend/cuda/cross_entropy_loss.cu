#include "cross_entropy_loss.h"

#include <cstdlib>

#include <cuda_fp16.h>

#include "reduction.cuh"
#include "utils.h"
#include "src/backend/utils.h"

namespace NeuroFrame::Backend::CUDA {

constexpr int64_t MAX_BLOCK_SIZE = 1024;

template<typename T>
static __device__ __forceinline__ T __exp_d(const T &val) {
	if constexpr (std::is_same_v<T, half>) {
		return __expf(val);
	} else if constexpr (std::is_same_v<T, float>) {
		return expf(val);
	} else if constexpr (std::is_same_v<T, double>) {
		return exp(val);
	}
}

template<typename T>
static __device__ __forceinline__ T __log_d(const T &val) {
	if constexpr (std::is_same_v<T, half>) {
		return __logf(val);
	} else if constexpr (std::is_same_v<T, float>) {
		return logf(val);
	} else if constexpr (std::is_same_v<T, double>) {
		return log(val);
	}
}

// batched_softmax_cross_entropy_loss_forward_kernel: Compute the softmax and apply cross entropy loss
// Grid size: (batch_size)
// Block size: (min(num_classes, MAX_BLOCK_SIZE))
template<typename T>
__global__ void batched_softmax_cross_entropy_loss_forward_kernel(
	T* __restrict__ loss_result,			// (batch_size)
	T* __restrict__ softmax_output,			// (batch_size, num_classes)
	const T* __restrict__ answer,	// (batch_size, num_classes)
	const int32_t* ground_truth,	// (batch_size)
	int64_t batch_size,
	int64_t num_classes
) {
	int64_t batch_id = blockIdx.x;
	const T* my_answer = answer + batch_id * num_classes;
	T* my_softmax_output = softmax_output + batch_id * num_classes;

	// Step 1: Calculate the output of the softmax

	// Step 1.1: Calculate the maximum value of the answer
	T local_max = get_min<T>();
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
		local_max = max(local_max, my_answer[i]);
	}
	T global_max = block_reduce_max_broadcast(local_max);

	// Step 1.2: Calculate the sum of the exponentials
	T local_sum = (T)0.0;
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
		T x = __exp_d(my_answer[i] - global_max);
		my_softmax_output[i] = x;
		local_sum += x;
	}
	T global_sum = block_reduce_sum_broadcast(local_sum);

	// Step 1.3: Calculate the softmax
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
		my_softmax_output[i] /= global_sum;
	}

	// Step 2. Calculate the cross entropy loss
	if (threadIdx.x == 0) {
		T x = my_softmax_output[ground_truth[batch_id]];
		const T thres = (T)1e-8;
		x = x < thres ? thres : x;
		loss_result[batch_id] = -__log_d(x);
	}
}

// batched_softmax_cross_entropy_loss_forward: Compute the softmax and apply cross entropy loss
// answer: The answer of the network, (batch_size, num_classes)
// ground_truth: The ground truth of the network, (batch_size)
// Return: <loss_result, softmax_output>
std::pair<Tensor, Tensor> batched_softmax_cross_entropy_loss_forward(const Tensor& answer, const Tensor& ground_truth) {
	if (ground_truth.dtype != dtype_t::INT32) {
		LOG_FATAL("ground_truth must be of type INT32");
	}
	if (answer.dim() != 2) {
		LOG_FATAL("answer must be of dimension 2");
	}
	if (ground_truth.dim() != 1) {
		LOG_FATAL("ground_truth must be of dimension 1");
	}
	int64_t batch_size = answer.shape[0];
	int64_t num_classes = answer.shape[1];

	Tensor loss_result = Tensor({batch_size}, answer.dtype, answer.device);
	Tensor softmax_output({batch_size, num_classes}, answer.dtype, answer.device);

	int64_t block_size = std::min(num_classes, MAX_BLOCK_SIZE);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(answer.dtype, batched_softmax_cross_entropy_loss_forward_kernel<<<batch_size, block_size>>>(
		(T*) loss_result.data_ptr(),
		(T*) softmax_output.data_ptr(),
		(const T*) answer.data_ptr(),
		(const int32_t*) ground_truth.data_ptr(),
		batch_size,
		num_classes
	));

	return {loss_result, softmax_output};
}

// batched_softmax_cross_entropy_loss_backward_kernel: Compute the gradient of the softmax and cross entropy loss
// Grid size: (batch_size)
// Block size: (min(num_classes, MAX_BLOCK_SIZE))
template<typename T>
__global__ void batched_softmax_cross_entropy_loss_backward_kernel(
	T* __restrict__ input_grad,			// (batch_size, num_classes)
	const T* __restrict__ output_grad,	// (batch_size)
	const T* __restrict__ softmax_output,	// (batch_size, num_classes)
	const int32_t* ground_truth,	// (batch_size)
	int64_t batch_size,
	int64_t num_classes
) {
	int64_t batch_id = blockIdx.x;
	T my_output_grad = output_grad[batch_id];
	const T* my_softmax_output = softmax_output + batch_id * num_classes;
	T* my_input_grad = input_grad + batch_id * num_classes;

	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
		my_input_grad[i] = my_softmax_output[i] * my_output_grad;
	}

	if (threadIdx.x == 0) {
		my_input_grad[ground_truth[batch_id]] -= (T)1.0*my_output_grad;
	}
}

Tensor batched_softmax_cross_entropy_loss_backward(const Tensor &output_grad, const Tensor &ground_truth, const Tensor &softmax_result) {
	Tensor input_grad(softmax_result.shape, softmax_result.dtype, softmax_result.device);

	int64_t batch_size = softmax_result.shape[0];
	int64_t num_classes = softmax_result.shape[1];

	int64_t block_size = std::min(num_classes, MAX_BLOCK_SIZE);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(softmax_result.dtype, batched_softmax_cross_entropy_loss_backward_kernel<<<batch_size, block_size>>>(
		(T*) input_grad.data_ptr(),
		(const T*) output_grad.data_ptr(),
		(const T*) softmax_result.data_ptr(),
		(const int32_t*) ground_truth.data_ptr(),
		batch_size,
		num_classes
	));
	return input_grad;
}

}