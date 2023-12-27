#include "batch_norm.h"

#include <iostream>

#include "utils.h"
#include "reduction.cuh"

namespace NeuroFrame::Backend::CUDA {

// batch_norm_forward_kernel
// input: (N, C, H, W)
// grid_dim: (C)
// block_dim: min(1024, N*H*W)
template<typename T>
__global__ void batch_norm_forward_kernel(
	T* __restrict__ output,	// (N, C, H, W)
	const T* __restrict__ input,	// (N, C, H, W)
	T* __restrict__ running_mean,			// (C)
	T* __restrict__ running_var,			// (C)
	T* __restrict__ sample_mean,			// (C)
	T* __restrict__ sample_var,			// (C)
	const T* __restrict__ gamma,	// (C)
	const T* __restrict__ beta,		// (C)
	const T momentum,
	const T eps,
	const int64_t N,
	const int64_t HW,
	bool update_stats
) {
	const int64_t C = gridDim.x;
	const int64_t my_channel = blockIdx.x;

	T mean_to_use, var_to_use;
	if (update_stats) {
		// Compute sample mean and sample variance
		T my_sum = 0., my_sum2 = 0.;
		#pragma unroll 4
		for (int64_t i = threadIdx.x; i < N*HW; i += blockDim.x) {
			int64_t index = INDEX_3D(N, C, HW, i/HW, my_channel, i%HW);
			const T x = input[index];
			my_sum += x;
			my_sum2 += x * x;
		}
		my_sum = block_reduce_sum(my_sum);
		my_sum2 = block_reduce_sum(my_sum2);

		__shared__ T s_sample_mean, s_sample_var;
		if (threadIdx.x == 0) {
			s_sample_mean = my_sum / (T)(long long)(N*HW);
			s_sample_var = my_sum2 / (T)(long long)(N*HW) - s_sample_mean * s_sample_mean;
			sample_mean[my_channel] = s_sample_mean;
			sample_var[my_channel] = s_sample_var;
			running_mean[my_channel] = momentum * running_mean[my_channel] + ((T)1. - momentum) * s_sample_mean;
			running_var[my_channel] = momentum * running_var[my_channel] + ((T)1. - momentum) * s_sample_var;
		}
		__syncthreads();
		mean_to_use = s_sample_mean;
		var_to_use = s_sample_var;
	} else {
		mean_to_use = running_mean[my_channel];
		var_to_use = running_var[my_channel];
	}

	const T var_invsqrt = rsqrt(var_to_use + eps);
	const T my_gamma = gamma[my_channel];
	const T my_beta = beta[my_channel];
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < N*HW; i += blockDim.x) {
		int64_t index = INDEX_3D(N, C, HW, i/HW, my_channel, i%HW);
		const T x = input[index];
		output[index] = my_gamma * (x - mean_to_use) * var_invsqrt + my_beta;
	}
}

// Return: output, sample_mean, sample_var
std::tuple<Tensor, Tensor, Tensor> batch_norm(
	const Tensor &input,
	const Tensor &gamma,
	const Tensor &beta,
	Tensor &running_mean,
	Tensor &running_var,
	double momentum,
	double eps,
	bool is_training
) {
	if (input.dim() != 4) {
		throw std::runtime_error("batch_norm: input must be 4D");
	}
	if (gamma.dim() != 1) {
		throw std::runtime_error("batch_norm: gamma must be 1D");
	}
	if (beta.dim() != 1) {
		throw std::runtime_error("batch_norm: beta must be 1D");
	}
	if (running_mean.dim() != 1) {
		throw std::runtime_error("batch_norm: running_mean must be 1D");
	}
	if (running_var.dim() != 1) {
		throw std::runtime_error("batch_norm: running_var must be 1D");
	}

	const int64_t N = input.shape[0];
	const int64_t C = input.shape[1];
	const int64_t H = input.shape[2];
	const int64_t W = input.shape[3];
	const int64_t HW = H * W;

	if (gamma.shape[0] != C) {
		throw std::runtime_error("batch_norm: gamma must have same size as input channels");
	}
	if (beta.shape[0] != C) {
		throw std::runtime_error("batch_norm: beta must have same size as input channels");
	}
	if (running_mean.shape[0] != C) {
		throw std::runtime_error("batch_norm: running_mean must have same size as input channels");
	}
	if (running_var.shape[0] != C) {
		throw std::runtime_error("batch_norm: running_var must have same size as input channels");
	}

	Tensor output = Tensor(input.shape, input.dtype, input.device);
	Tensor sample_mean = Tensor({C}, input.dtype, input.device);
	Tensor sample_var = Tensor({C}, input.dtype, input.device);

	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype,
		batch_norm_forward_kernel<T><<<C, std::min(1024L, N*HW)>>>(
			(T*) output.data_ptr(),
			(const T*) input.data_ptr(),
			(T*) running_mean.data_ptr(),
			(T*) running_var.data_ptr(),
			(T*) sample_mean.data_ptr(),
			(T*) sample_var.data_ptr(),
			(const T*) gamma.data_ptr(),
			(const T*) beta.data_ptr(),
			(T) momentum,
			(T) eps,
			N,
			HW,
			is_training
		);
	);

	return {output, sample_mean, sample_var};
}

template<typename T>
__device__ __forceinline__ T my_sqrt(T x) {
	if constexpr(std::is_same_v<T, float>) {
		return sqrtf(x);
	} else if constexpr(std::is_same_v<T, double>){
		return sqrt(x);
	} else if constexpr(std::is_same_v<T, half>) {
		return hsqrt(x);
	} else {
		assert(false);
	}
}

// grid_shape: (C)
// block_shape: min(1024, N*H*W)
template<typename T>
__global__ void batch_norm_backward_kernel(
	T* __restrict__ input_grad,	// (N, C, H, W)
	T* __restrict__ gamma_grad,	// (C)
	const T* __restrict__ output_grad,	// (N, C, H, W)
	const T* __restrict__ input,	// (N, C, H, W)
	const T* __restrict__ gamma,	// (C)
	const T* __restrict__ sample_mean,	// (C)
	const T* __restrict__ sample_var,	// (C)
	int64_t N,
	int64_t HW
) {
	const int64_t C = gridDim.x;
	const int64_t my_c = blockIdx.x;
	const T my_gamma = gamma[my_c];
	const T my_sample_mean = sample_mean[my_c];
	const T my_sample_var = sample_var[my_c];

	const T var_invsqrt = rsqrt(my_sample_var + (T)1e-5);
	const T var_inv = (T)1. / (my_sample_var + (T)1e-5);
	const T NHW_inv = (T)1. / (T)(long long)(N * HW);

	T my_gamma_grad = (T)0.;
	T my_grad_sum = (T)0.;
	T my_grad_mul_delta_sum = (T)0.;
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < N*HW; i += blockDim.x) {
		int64_t index = INDEX_3D(0, C, HW, i/HW, my_c, i%HW);
		const T x = input[index];
		const T dy = output_grad[index];
		my_grad_sum += dy;
		my_gamma_grad += dy * (x - my_sample_mean) * var_invsqrt;
		my_grad_mul_delta_sum += dy * (x - my_sample_mean);
	}
	my_gamma_grad = block_reduce_sum(my_gamma_grad);
	if (threadIdx.x == 0) {
		gamma_grad[my_c] = my_gamma_grad;
	}
	my_grad_sum = block_reduce_sum_broadcast(my_grad_sum);
	my_grad_mul_delta_sum = block_reduce_sum_broadcast(my_grad_mul_delta_sum);
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < N*HW; i += blockDim.x) {
		int64_t index = INDEX_3D(0, C, HW, i/HW, my_c, i%HW);
		const T x = input[index];
		const T dy = output_grad[index];
		input_grad[index] = my_gamma * var_invsqrt * (dy - NHW_inv * my_grad_sum - (x-my_sample_mean)*var_inv*NHW_inv*my_grad_mul_delta_sum);
	}
}

// Return: input_grad, gamma_grad
std::tuple<Tensor, Tensor> batch_norm_backward(
	const Tensor &output_grad,
	const Tensor &input,
	const Tensor &gamma,
	const Tensor &sample_mean,
	const Tensor &sample_var
) {
	const int64_t N = input.shape[0];
	const int64_t C = input.shape[1];
	const int64_t H = input.shape[2];
	const int64_t W = input.shape[3];

	Tensor input_grad = Tensor(input.shape, input.dtype, input.device);
	Tensor gamma_grad = Tensor::zeros_like(gamma);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype,
		batch_norm_backward_kernel<T><<<dim3(C), std::min(1024L, N*H*W)>>>(
			(T*) input_grad.data_ptr(),
			(T*) gamma_grad.data_ptr(),
			(const T*) output_grad.data_ptr(),
			(const T*) input.data_ptr(),
			(const T*) gamma.data_ptr(),
			(const T*) sample_mean.data_ptr(),
			(const T*) sample_var.data_ptr(),
			N,
			H*W
		);
	);
	return {input_grad, gamma_grad};
}

// Grid shape: (N, C)
// Block shape: min(1024, N*H*W)
template<typename T>
__global__ void batch_norm_beta_grad_kernel(
	T* __restrict__ beta_grad,	// (C)
	const T* __restrict__ output_grad,	// (N, C, H, W)
	int64_t HW
) {
	const int64_t C = gridDim.y;
	const int64_t my_n = blockIdx.x;
	const int64_t my_c = blockIdx.y;

	T my_beta_grad = (T)0.;
	#pragma unroll 4
	for (int64_t i = threadIdx.x; i < HW; i += blockDim.x) {
		int64_t index = INDEX_3D(0, C, HW, my_n, my_c, i);
		my_beta_grad += output_grad[index];
	}
	my_beta_grad = block_reduce_sum(my_beta_grad);
	if (threadIdx.x == 0) {
		// TODO Optimize this. We may reduce the number of grids
		atomicAdd(beta_grad + my_c, my_beta_grad);
	}
}

// Return: beta_grad
Tensor batch_norm_beta_grad(
	const Tensor &output_grad
) {
	const int64_t N = output_grad.shape[0];
	const int64_t C = output_grad.shape[1];
	const int64_t H = output_grad.shape[2];
	const int64_t W = output_grad.shape[3];

	Tensor beta_grad = Tensor::zeros({C}, output_grad.dtype, output_grad.device);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(output_grad.dtype,
		batch_norm_beta_grad_kernel<T><<<dim3(N, C), std::min(1024L, N*H*W)>>>(
			(T*) beta_grad.data_ptr(),
			(const T*) output_grad.data_ptr(),
			H*W
		);
	);
	return beta_grad;
}

}