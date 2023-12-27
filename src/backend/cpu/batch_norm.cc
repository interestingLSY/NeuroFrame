#include "batch_norm.h"

#include <cmath>

#include "utils.h"

namespace NeuroFrame::Backend::CPU {

// batch_norm_forward_kernel
// input: (N, C, H, W)
template<typename T>
void batch_norm_forward_kernel(
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
	const int64_t C,
	const int64_t HW,
	bool update_stats
) {
	#pragma omp parallel for schedule(static)
	for (int64_t my_channel = 0; my_channel < C; ++my_channel) {
		T mean_to_use, var_to_use;
		if (update_stats) {
			// Compute sample mean and sample variance
			T sample_sum = 0., sample_sum2 = 0.;
			for (int64_t i = 0; i < N*HW; ++i) {
				const int64_t n = i / HW;
				const int64_t hw = i % HW;
				const T x = input[INDEX_3D(N, C, HW, n, my_channel, hw)];
				sample_sum = sample_sum + x;
				sample_sum2 = sample_sum2 + x * x;
			}
			sample_mean[my_channel] = sample_sum / (N*HW);
			sample_var[my_channel] = sample_sum2 / (N*HW) - sample_mean[my_channel] * sample_mean[my_channel];
			
			running_mean[my_channel] = momentum * running_mean[my_channel] + ((T)1. - momentum) * sample_mean[my_channel];
			running_var[my_channel] = momentum * running_var[my_channel] + ((T)1. - momentum) * sample_var[my_channel];

			mean_to_use = sample_mean[my_channel];
			var_to_use = sample_var[my_channel];
		} else {
			mean_to_use = running_mean[my_channel];
			var_to_use = running_var[my_channel];
		}

		const T var_inv = (T)(1.0f / std::sqrt((float)(var_to_use + eps)));
		const T my_gamma = gamma[my_channel];
		const T my_beta = beta[my_channel];

		for (int64_t i = 0; i < N*HW; ++i) {
			int64_t index = INDEX_3D(N, C, HW, i/HW, my_channel, i%HW);
			const T x = input[index];
			output[index] = my_gamma * (x - mean_to_use) * var_inv + my_beta;
		}
	}
}

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

	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype,
		batch_norm_forward_kernel<T>(
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
			C,
			HW,
			is_training
		);
	);

	return {output, sample_mean, sample_var};
}

template<typename T>
void batch_norm_backward_kernel(
	T* __restrict__ input_grad,	// (N, C, H, W)
	T* __restrict__ gamma_grad,	// (C)
	const T* __restrict__ output_grad,	// (N, C, H, W)
	const T* __restrict__ input,	// (N, C, H, W)
	const T* __restrict__ gamma,	// (C)
	const T* __restrict__ sample_mean,	// (C)
	const T* __restrict__ sample_var,	// (C)
	int64_t N,
	int64_t C,
	int64_t HW
) {
	#pragma omp parallel for schedule(static)
	for (int64_t my_c = 0; my_c < C; ++my_c) {
		const T my_gamma = gamma[my_c];
		const T my_sample_mean = sample_mean[my_c];
		const T my_sample_var = sample_var[my_c];

		const T var_invsqrt = (T)(1.0f / std::sqrt((float)(my_sample_var + (T)1e-5)));
		const T var_inv = (T)1. / (my_sample_var + (T)1e-5);
		const T NHW_inv = (T)1. / (N * HW);

		T cur_gamma_grad = (T)0.;
		T my_grad_sum = (T)0.;
		T my_grad_mul_delta_sum = (T)0.;

		for (int64_t i = 0; i < N*HW; i++) {
			int64_t index = INDEX_3D(N, C, HW, i/HW, my_c, i%HW);
			const T x = input[index];
			const T dy = output_grad[index];
			cur_gamma_grad = cur_gamma_grad + dy * (x - my_sample_mean) * var_invsqrt;
			my_grad_sum = my_grad_sum + dy;
			my_grad_mul_delta_sum = my_grad_mul_delta_sum + dy * (x - my_sample_mean);
		}
		gamma_grad[my_c] = cur_gamma_grad;
		for (int64_t i = 0; i < N*HW; i++) {
			int64_t index = INDEX_3D(N, C, HW, i/HW, my_c, i%HW);
			const T x = input[index];
			const T dy = output_grad[index];
			input_grad[index] = my_gamma * var_invsqrt * (dy - NHW_inv * my_grad_sum - (x-my_sample_mean)*var_inv*NHW_inv*my_grad_mul_delta_sum);
		}
	}
}

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
	Tensor gamma_grad = Tensor(gamma.shape, gamma.dtype, gamma.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype,
		batch_norm_backward_kernel<T>(
			(T*) input_grad.data_ptr(),
			(T*) gamma_grad.data_ptr(),
			(const T*) output_grad.data_ptr(),
			(const T*) input.data_ptr(),
			(const T*) gamma.data_ptr(),
			(const T*) sample_mean.data_ptr(),
			(const T*) sample_var.data_ptr(),
			N,
			C,
			H*W
		);
	);
	return {input_grad, gamma_grad};
}

template<typename T>
void batch_norm_beta_grad_kernel(
	T* __restrict__ beta_grad,	// (C)
	const T* __restrict__ output_grad,	// (N, C, H, W)
	int64_t N,
	int64_t C,
	int64_t HW
) {
	#pragma omp parallel for schedule(static)
	for (int64_t my_c = 0; my_c < C; ++my_c) {
		T cur_beta_grad = (T)0.;
		for (int64_t my_n = 0; my_n < N; ++my_n) {
			for (int64_t i = 0; i < HW; i++) {
				int64_t index = INDEX_3D(N, C, HW, my_n, my_c, i);
				const T dy = output_grad[index];
				cur_beta_grad = cur_beta_grad + dy;
			}
		}
		beta_grad[my_c] = cur_beta_grad;
	}
}

Tensor batch_norm_beta_grad(
	const Tensor &output_grad
) {
	const int64_t N = output_grad.shape[0];
	const int64_t C = output_grad.shape[1];
	const int64_t H = output_grad.shape[2];
	const int64_t W = output_grad.shape[3];

	Tensor beta_grad = Tensor({C}, output_grad.dtype, output_grad.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(output_grad.dtype,
		batch_norm_beta_grad_kernel<T>(
			(T*) beta_grad.data_ptr(),
			(const T*) output_grad.data_ptr(),
			N,
			C,
			H*W
		);
	);
	return beta_grad;
}

}