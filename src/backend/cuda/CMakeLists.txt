add_library(neuroframe_backend_cuda STATIC
	relu.cu
	sigmoid.cu
	tensor_eq.cu
	transpose.cu
	matmul.cu
	tensor_binary_op.cu
	tensor_unary_op.cu
	pool.cu
	cross_entropy_loss.cu
	convolution.cu
)

target_link_libraries(neuroframe_backend_cuda PUBLIC neuroframe_tensor CUDA::cudart CUDA::cublas)