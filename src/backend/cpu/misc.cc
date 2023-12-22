#include "misc.h"

#include <cstdint>
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

}
