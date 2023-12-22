#pragma once

// reduction.cuh: CUDA reduction operations

#include <cuda_runtime.h>

#include "src/backend/utils.h"
#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

#define WARP_SIZE 32

// warp_reduce_XXX: Reduction within a warp
// After this operation, every thread in the warp will have the same value

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
	#pragma unroll
	for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
		val += __shfl_xor_sync(0xffffffff, val, offset);
	}
	return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
	#pragma unroll
	for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
		val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
	}
	return val;
}

// block_reduce_XXX: Reduction within the whole block
// After this operation, only the first thread in the block will have the value

template<typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
	static __shared__ T shared[32];	// 32 is the maximum number of warps in a block
	int lane = threadIdx.x % WARP_SIZE;
	int wid = threadIdx.x / WARP_SIZE;
	val = warp_reduce_sum(val);
	if (lane == 0) shared[wid] = val;
	__syncthreads();
	
	if (wid == 0) {
		val = (threadIdx.x*WARP_SIZE <= blockDim.x-1) ? shared[lane] : (T)0.0;
		val = warp_reduce_sum(val);
	}
	return val;
}

template<typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
	static __shared__ T shared[32];
	int lane = threadIdx.x % WARP_SIZE;
	int wid = threadIdx.x / WARP_SIZE;
	val = warp_reduce_max(val);
	if (lane == 0) shared[wid] = val;
	__syncthreads();

	if (wid == 0) {
		val = (threadIdx.x*WARP_SIZE <= blockDim.x-1) ? shared[lane] : get_min<T>();
		val = warp_reduce_max(val);
	}
	return val;
}

// block_reduce_XXX_broadcast: Reduction within the whole block
// After this operation, all threads in the block will have the value

template<typename T>
__device__ __forceinline__ T block_reduce_sum_broadcast(T val) {
	static __shared__ T shared[32];
	int lane = threadIdx.x % WARP_SIZE;
	int wid = threadIdx.x / WARP_SIZE;
	val = warp_reduce_sum(val);
	if (lane == 0) shared[wid] = val;
	__syncthreads();
	
	if (wid == 0) {
		val = (threadIdx.x*WARP_SIZE <= blockDim.x-1) ? shared[lane] : (T)0.0;
		val = warp_reduce_sum(val);
		shared[lane] = val;	// Save to shared[lane] to avoid bank conflict
	}
	__syncthreads();

	val = shared[0];
	return val;
}

template<typename T>
static __device__ __forceinline__ T block_reduce_max_broadcast(T val) {
	static __shared__ T shared[32];
	int lane = threadIdx.x % WARP_SIZE;
	int wid = threadIdx.x / WARP_SIZE;
	val = warp_reduce_max(val);
	if (lane == 0) shared[wid] = val;
	__syncthreads();
	
	if (wid == 0) {
		val = (threadIdx.x*WARP_SIZE <= blockDim.x-1) ? shared[lane] : get_min<T>();
		val = warp_reduce_max(val);
		shared[lane] = val;
	}
	__syncthreads();

	val = shared[0];
	return val;
}

#undef WARP_SIZE

}	// namespace NeuroFrame::Backend::CUDA
