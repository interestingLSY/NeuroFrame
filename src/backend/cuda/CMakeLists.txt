add_library(neuroframe_backend_cuda STATIC
	broadcast.cu
	convolution.cu
	cross_entropy_loss.cu
	matmul.cu
	misc.cu
	pool.cu
	relu.cu
	sigmoid.cu
	tensor_eq.cu
	tensor_binary_op.cu
	tensor_reduction_op.cu
	tensor_scalar_op.cu
	tensor_unary_op.cu
	transpose.cu
)

target_link_libraries(neuroframe_backend_cuda PUBLIC neuroframe_tensor CUDA::cudart CUDA::cublas)