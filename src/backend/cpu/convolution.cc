#include "convolution.h"

#include "src/basic/log.h"
#include "src/tensor/tensor.h"

#include "utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T>
void batched_im2col_kernel(
	T* __restrict__ output,			// (batch_size, h*w, c_in*kh*kw)
	const T* __restrict__ input,	// (batch_size, c_in, h, w)
	const int64_t batch_size,
	const int64_t c_in,
	const int64_t h,
	const int64_t w,
	const int64_t kh,
	const int64_t kw
) {
	#pragma omp parallel for collapse(4) schedule(static)
	for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
		for (int64_t c_in_index = 0; c_in_index < c_in; ++c_in_index) {
			for (int64_t h_index = 0; h_index < h; ++h_index) {
				for (int64_t w_index = 0; w_index < w; ++w_index) {
					int64_t output_index = batch_index * h * w * c_in * kh * kw
						+ h_index * w * c_in * kh * kw
						+ w_index * c_in * kh * kw
						+ c_in_index * kh * kw;
					for (int64_t kh_index = 0; kh_index < kh; ++kh_index) {
						for (int64_t kw_index = 0; kw_index < kw; ++kw_index) {
							int64_t input_index = batch_index * c_in * h * w
								+ c_in_index * h * w
								+ (h_index + kh_index - kh / 2) * w
								+ (w_index + kw_index - kw / 2);
							if (h_index + kh_index - kh / 2 < 0
								|| h_index + kh_index - kh / 2 >= h
								|| w_index + kw_index - kw / 2 < 0
								|| w_index + kw_index - kw / 2 >= w) {
								output[output_index] = (T)0.;
							} else {
								output[output_index] = input[input_index];
							}
							++output_index;
						}
					}
				}
			}
		}
	}
}

// batched_im2col: im2col with padding = 0 (zero padding), stride = 1
Tensor batched_im2col(const Tensor &input_img, const int64_t kh, const int64_t kw) {
	if (input_img.dim() != 4) {
		LOG_FATAL("input_img.dim() != 4");
	}
	if (kh % 2 == 0 || kw % 2 == 0) {
		LOG_FATAL("kh or kw is even");
	}
	int64_t batch_size = input_img.shape[0];
	int64_t c_in = input_img.shape[1];
	int64_t h = input_img.shape[2];
	int64_t w = input_img.shape[3];

	Tensor result({batch_size, h*w, c_in*kh*kw}, input_img.dtype, input_img.device);

	DISPATCH_ON_DTYPE_CPU_BACKEND(input_img.dtype, batched_im2col_kernel(
		(T*) result.data_ptr(),
		(const T*) input_img.data_ptr(),
		batch_size,
		c_in,
		h,
		w,
		kh,
		kw
	));

	return result;
}

template<typename T>
void batched_col2im_kernel(
	T* __restrict__ output,				// (batch_size, c_in, h, w)
	const T* __restrict__ col_grads,	// (batch_size, h*w, c_in*kh*kw)
	const int64_t batch_size,
	const int64_t c_in,
	const int64_t h,
	const int64_t w,
	const int64_t kh,
	const int64_t kw
) {
	#pragma omp parallel for collapse(2)
	for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
		for (int64_t c_in_index = 0; c_in_index < c_in; ++c_in_index) {
			for (int64_t h_index = 0; h_index < h; ++h_index) {
				for (int64_t w_index = 0; w_index < w; ++w_index) {
					for (int64_t kh_index = 0; kh_index < kh; ++kh_index) {
						for (int64_t kw_index = 0; kw_index < kw; ++kw_index) {
							int64_t origin_h_index = h_index + kh_index - kh/2;
							int64_t origin_w_index = w_index + kw_index - kw/2;
							if (origin_h_index >= 0 && origin_h_index < h
								&& origin_w_index >= 0 && origin_w_index < w) {
								int64_t output_index = batch_index * c_in * h * w
									+ c_in_index * h * w
									+ origin_h_index * w
									+ origin_w_index;
								int64_t col_grads_index = batch_index * h * w * c_in * kh * kw
									+ h_index * w * c_in * kh * kw
									+ w_index * c_in * kh * kw
									+ c_in_index * kh * kw
									+ kh_index * kw
									+ kw_index;
								output[output_index] = output[output_index] + col_grads[col_grads_index];
							}
						}
					}
				}
			}
		}
	}
}

// input: (batch_size, h*w, c_in*kh*kw)
Tensor batched_col2im(const Tensor &input, const int64_t c_in, const int64_t h, const int64_t w, const int64_t kh, const int64_t kw) {
	if (input.dim() != 3) {
		LOG_FATAL("input.dim() != 3");
	}
	if (kh % 2 == 0 || kw % 2 == 0) {
		LOG_FATAL("kh or kw is even");
	}

	int64_t batch_size = input.shape[0];

	Tensor result = Tensor::zeros({batch_size, c_in, h, w}, input.dtype, input.device);

	DISPATCH_ON_DTYPE_CPU_BACKEND(
		input.dtype,
		batched_col2im_kernel(
			(T*) result.data_ptr(),
			(const T*) input.data_ptr(),
			batch_size,
			c_in,
			h,
			w,
			kh,
			kw
	));

	return result;
}

}