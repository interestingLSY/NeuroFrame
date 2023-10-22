#include "cross_entropy_loss.h"

namespace NeuroFrame::Backend::CUDA {

template<typename T>
void batched_softmax_cross_entropy_loss_kernel(
	const T* answer,

) {

}

// batched_softmax_cross_entropy_loss: Compute the softmax and apply cross entropy loss
// answer: The answer of the network, (batched_size, num_classes)
// ground_truth: The ground truth of the network, (batched_size)
Tensor batched_softmax_cross_entropy_loss(const Tensor& answer, const Tensor& ground_truth) {

}

}