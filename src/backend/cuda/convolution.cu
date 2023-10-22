#include "convolution.h"

#include "src/basic/log.h"
#include "src/tensor/tensor.h"

#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

static constexpr int64_t MAX_BLOCK_SIZE = 1024;

// batched_im2col_kernel
// grid_size: (batch_size, c_in, h)
// block_size: (min(w, MAX_BLOCK_SIZE))
// Each thread is responsible for distributing one pixel into the output tensor
template<typename T>
__global__ void batched_im2col_kernel(
	T* __restrict__ output,			// (batch_size, h*w, c_in*kh*kw)
	const T* __restrict__ input,	// (batch_size, c_in, h, w)
	const int64_t batch_size,
	const int64_t c_in,
	const int64_t h,
	const int64_t w,
	const int64_t kh,
	const int64_t kw
) {
	int64_t batch_index = blockIdx.x;
	int64_t c_in_index = blockIdx.y;
	int64_t h_index = blockIdx.z;

	for (int64_t w_index = threadIdx.x; w_index < w; w_index += blockDim.x) {
		T* cur_output = output + batch_index*(h*w*c_in*kh*kw) + h_index*(w*c_in*kh*kw) + w_index*(c_in*kh*kw) + c_in_index*(kh*kw);
		const T* cur_input = input + batch_index*(c_in*h*w) + c_in_index*(h*w) + h_index*(w) + w_index;
		for (int64_t kh_index = -(kh/2); kh_index <= kh/2; ++kh_index) {
			for (int64_t kw_index = -(kw/2); kw_index <= kw/2; ++kw_index) {
				*cur_output = 
					h_index + kh_index < 0 || h_index + kh_index >= h ||
					w_index + kw_index < 0 || w_index + kw_index >= w ?
					(T)0. :
					cur_input[kh_index*w + kw_index];
				cur_output += 1;
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

	int64_t block_size = std::min(w, MAX_BLOCK_SIZE);
	dim3 grid_size(batch_size, c_in, h);

	DISPATCH_ON_DTYPE_CUDA_BACKEND(input_img.dtype, batched_im2col_kernel<<<grid_size, block_size>>>(
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

// batched_col2im_kernel
// grid_size: (batch_size, c_in, h)
// block_size: (min(w, MAX_BLOCK_SIZE))
// Each thread is responsible for reduction of one pixel in the output tensor
template<typename T>
__global__ void batched_col2im_kernel(
	T* __restrict__ output,				// (batch_size, c_in, h, w)
	const T* __restrict__ col_grads,	// (batch_size, h*w, c_in*kh*kw)
	const int64_t batch_size,
	const int64_t c_in,
	const int64_t h,
	const int64_t w,
	const int64_t kh,
	const int64_t kw
) {
	int64_t batch_index = blockIdx.x;
	int64_t c_in_index = blockIdx.y;
	int64_t h_index = blockIdx.z;

	for (int64_t w_index = threadIdx.x; w_index < w; w_index += blockDim.x) {
		T result = (T)0.;
		for (int64_t kh_index = -(kh/2); kh_index <= kh/2; ++kh_index) {
			for (int64_t kw_index = -(kw/2); kw_index <= kw/2; ++kw_index) {
				int64_t conv_center_h_index = h_index + kh_index;
				int64_t conv_center_w_index = w_index + kw_index;
				if (conv_center_h_index >= 0 && conv_center_h_index < h &&
					conv_center_w_index >= 0 && conv_center_w_index < w) {
					int64_t conv_relpos_h_index = kh/2 - kh_index;
					int64_t conv_relpos_w_index = kw/2 - kw_index;
					// printf("%ld %ld\n", conv_relpos_h_index, conv_relpos_w_index);
					int64_t col_index =
						batch_index * (h*w*c_in*kh*kw) +
						conv_center_h_index * (w*c_in*kh*kw) +
						conv_center_w_index * (c_in*kh*kw) +
						c_in_index * (kh*kw) +
						conv_relpos_h_index * (kw) +
						conv_relpos_w_index;
					T contrib = col_grads[col_index];
					result += contrib;
				}
			}
		}
		output[
			batch_index * (c_in*h*w) +
			c_in_index * (h*w) +
			h_index * (w) +
			w_index
		] = result;
	}
}

// input: (batch_size, h*w, c_in*kh*kw)
Tensor batched_col2im(const Tensor &input, const int64_t c_in, const int64_t h, const int64_t w, const int64_t kh, const int64_t kw) {
	if (input.dim() != 3) {
		LOG_FATAL("input.dim() != im2col_result_grad3");
	}
	if (kh % 2 == 0 || kw % 2 == 0) {
		LOG_FATAL("kh or kw is even");
	}

	int64_t batch_size = input.shape[0];

	Tensor result({batch_size, c_in, h, w}, input.dtype, input.device);

	int64_t block_size = std::min(w, MAX_BLOCK_SIZE);
	dim3 grid_size(batch_size, c_in, h);

	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, batched_col2im_kernel<<<grid_size, block_size>>>(
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