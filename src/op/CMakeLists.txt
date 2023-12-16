add_library(neuroframe_op STATIC
	op.cc
	op_context.cc
	op_utils.cc
	broadcast.cc
	relu.cc
	reshape.cc
	sigmoid.cc
	tensor_eq.cc
	transpose.cc
	matmul.cc
	tensor_binary_op.cc
	tensor_unary_op.cc
	pool.cc
	cross_entropy_loss.cc
	convolution.cc
)

target_link_libraries(neuroframe_op PUBLIC
	neuroframe_tensor
	neuroframe_backend_cpu
	neuroframe_backend_cuda)