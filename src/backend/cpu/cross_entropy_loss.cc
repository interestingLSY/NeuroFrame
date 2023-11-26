#include "cross_entropy_loss.h"

#include <cstdlib>
#include <cmath>

#include "utils.h"
#include "src/backend/utils.h"

namespace NeuroFrame::Backend::CPU {

// batched_softmax_cross_entropy_loss_forward_kernel: Compute the softmax and apply cross entropy loss
template<typename T>
void batched_softmax_cross_entropy_loss_forward_kernel(
	T* __restrict__ loss_result,			// ()
	T* __restrict__ softmax_output,			// (batch_size, num_classes)
	const T* __restrict__ answer,	// (batch_size, num_classes)
	const int32_t* ground_truth,	// (batch_size)
	int64_t batch_size,
	int64_t num_classes
) {
	typedef std::conditional_t<std::is_same_v<T, half>, float, T> reduction_t;	// Reduce in float if T is half since OMP does not support half reduction
	reduction_t my_loss_result = (reduction_t)0.;
	#pragma omp parallel for schedule(static) reduction(+:my_loss_result)
	for (int64_t batch_id = 0; batch_id < batch_size; ++batch_id) {
		const T* my_answer = answer + batch_id*num_classes;
		T* my_softmax_output = softmax_output + batch_id*num_classes;

		T max_val = get_min<T>();
		for (int64_t i = 0; i < num_classes; ++i) {
			if (my_answer[i] > max_val) {
				max_val = my_answer[i];
			}
		}

		T sum = (T)0.;
		for (int64_t i = 0; i < num_classes; ++i) {
			my_softmax_output[i] = std::exp(my_answer[i] - max_val);
			sum = sum + my_softmax_output[i];
		}

		#pragma omp simd
		for (int64_t i = 0; i < num_classes; ++i) {
			my_softmax_output[i] = my_softmax_output[i] / sum;
		}

		my_loss_result += (reduction_t)-std::log(my_softmax_output[ground_truth[batch_id]]);
	}
	*loss_result = (T)(my_loss_result / (reduction_t)batch_size);
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

	Tensor loss_result({}, answer.dtype, answer.device);
	Tensor softmax_output({batch_size, num_classes}, answer.dtype, answer.device);

	DISPATCH_ON_DTYPE_CPU_BACKEND(answer.dtype, batched_softmax_cross_entropy_loss_forward_kernel(
		(T*) loss_result.data_ptr(),
		(T*) softmax_output.data_ptr(),
		(const T*) answer.data_ptr(),
		(const int32_t*) ground_truth.data_ptr(),
		batch_size,
		num_classes
	));

	return {loss_result, softmax_output};
}

// batched_softmax_cross_entropy_loss_backward_kernel
template<typename T>
void batched_softmax_cross_entropy_loss_backward_kernel(
	T* __restrict__ input_grad,			// (batch_size, num_classes)
	const T* __restrict__ output_grad,	// (batch_size)
	const T* __restrict__ softmax_output,	// (batch_size, num_classes)
	const int32_t* ground_truth,	// (batch_size)
	int64_t batch_size,
	int64_t num_classes
) {
	#pragma omp parallel for schedule(static)
	for (int64_t batch_id = 0; batch_id < batch_size; ++batch_id) {
		T my_output_grad = output_grad[batch_id];
		const T* my_softmax_output = softmax_output + batch_id*num_classes;
		T* my_input_grad = input_grad + batch_id*num_classes;

		#pragma omp simd
		for (int64_t i = 0; i < num_classes; ++i) {
			my_input_grad[i] = my_softmax_output[i] * my_output_grad;
		}

		my_input_grad[ground_truth[batch_id]] = my_input_grad[ground_truth[batch_id]] - my_output_grad;
	}
}

Tensor batched_softmax_cross_entropy_loss_backward(const Tensor &output_grad, const Tensor &ground_truth, const Tensor &softmax_result) {
	Tensor input_grad(softmax_result.shape, softmax_result.dtype, softmax_result.device);

	int64_t batch_size = softmax_result.shape[0];
	int64_t num_classes = softmax_result.shape[1];

	DISPATCH_ON_DTYPE_CPU_BACKEND(output_grad.dtype, batched_softmax_cross_entropy_loss_backward_kernel(
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
