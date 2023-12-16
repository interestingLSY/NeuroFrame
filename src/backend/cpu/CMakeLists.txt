add_library(neuroframe_backend_cpu STATIC
	broadcast.cc
	relu.cc
	sigmoid.cc
	tensor_eq.cc
	transpose.cc
	matmul.cc
	tensor_binary_op.cc
	tensor_scalar_op.cc
	tensor_unary_op.cc
	pool.cc
	cross_entropy_loss.cc
	convolution.cc
)

target_link_libraries(neuroframe_backend_cpu PUBLIC neuroframe_tensor)