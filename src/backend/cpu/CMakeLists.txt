add_library(neuroframe_backend_cpu STATIC
	batch_norm.cc
	broadcast.cc
	convolution.cc
	cross_entropy_loss.cc
	matmul.cc
	misc.cc
	pool.cc
	relu.cc
	sigmoid.cc
	tensor_eq.cc
	tensor_binary_op.cc
	tensor_reduction_op.cc
	tensor_scalar_op.cc
	tensor_unary_op.cc
	transpose.cc
)

target_link_libraries(neuroframe_backend_cpu PUBLIC neuroframe_tensor)