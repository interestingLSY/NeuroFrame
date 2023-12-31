#include "misc.h"

#include <cstdint>
#include <cmath>
#include <omp.h>

#include "utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T>
int get_correct_sample_count_kernel(
	const T* __restrict__ output,	// [batch_size, num_classes]
	const int32_t* __restrict__ ground_truth,	// [batch_size]
	int64_t batch_size,
	int64_t num_classes
) {
	int64_t correct = 0;
	#pragma omp parallel for reduction(+:correct)
	for (int64_t i = 0; i < batch_size; i++) {
		int32_t pred_class = 0;
		T max = output[i * num_classes];
		for (int64_t j = 1; j < num_classes; j++) {
			if (output[i * num_classes + j] > max) {
				max = output[i * num_classes + j];
				pred_class = j;
			}
		}
		if (pred_class == ground_truth[i]) {
			correct++;
		}
	}
	return correct;
}

int64_t get_correct_sample_count(const Tensor &output, const Tensor &ground_truth) {
	int64_t batch_size = output.shape[0];
	int64_t num_classes = output.shape[1];
	int64_t answer = DISPATCH_ON_DTYPE_CPU_BACKEND(
		output.dtype,
		get_correct_sample_count_kernel(
			(const T*) output.data_ptr(),
			(const int32_t*) ground_truth.data_ptr(),
			batch_size,
			num_classes
		)
	);
	return answer;
}

template<typename T>
void sgd_grad_update_kernel(
	T* __restrict__ weight,
	const T* __restrict__ grad,
	T* __restrict__ momentum,
	T learning_rate,
	T momentum_factor,
	T weight_decay,
	int64_t num_elements
) {
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < num_elements; i++) {
		T cur_grad = weight_decay != 0 ? (T)(grad[i] + weight_decay * weight[i]) : grad[i];
		if (momentum_factor != 0) {
			cur_grad = momentum[i] = momentum_factor * momentum[i] + cur_grad;
		}
		weight[i] = weight[i] - learning_rate*cur_grad;
	}
}

void sgd_grad_update(Tensor &weight, const Tensor &grad, Tensor &momentum, double learning_rate, double momentum_factor, double weight_decay) {
	int64_t num_elements = weight.numel();
	DISPATCH_ON_DTYPE_CPU_BACKEND(
		weight.dtype,
		sgd_grad_update_kernel(
			(T*) weight.data_ptr(),
			(const T*) grad.data_ptr(),
			(T*) momentum.data_ptr(),
			(T) learning_rate,
			(T) momentum_factor,
			(T) weight_decay,
			num_elements
		)
	);
}

template<typename T>
void adam_grad_update_kernel(
	T* __restrict__ weight,
	const T* __restrict__ grad,
	T* __restrict__ momentum,
	T* __restrict__ velocity,
	T learning_rate,
	T beta1,
	T beta2,
	T eps,
	int64_t cur_timestep,
	int64_t num_elements
) {
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < num_elements; i++) {
		T cur_grad = grad[i];
		momentum[i] = beta1 * momentum[i] + (1.0 - beta1) * cur_grad;
		velocity[i] = beta2 * velocity[i] + (1.0 - beta2) * cur_grad * cur_grad;
		T adjusted_momentum = momentum[i] / (1.0 - std::pow(beta1, cur_timestep));
		T adjusted_velocity = velocity[i] / (1.0 - std::pow(beta2, cur_timestep));
		weight[i] = weight[i] - learning_rate * adjusted_momentum / (std::sqrt(adjusted_velocity) + eps);
	}
}

void adam_grad_update(Tensor &weight, const Tensor &grad, Tensor &momentum, Tensor &velocity, int cur_timestamp, double learning_rate, double beta1, double beta2, double eps) {
	int64_t num_elements = weight.numel();
	DISPATCH_ON_DTYPE_CPU_BACKEND(
		weight.dtype,
		adam_grad_update_kernel(
			(T*) weight.data_ptr(),
			(const T*) grad.data_ptr(),
			(T*) momentum.data_ptr(),
			(T*) velocity.data_ptr(),
			(T) learning_rate,
			(T) beta1,
			(T) beta2,
			(T) eps,
			cur_timestamp,
			num_elements
		)
	);
}

}
