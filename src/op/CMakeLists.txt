add_library(neuroframe_op STATIC
	op.cc
	op_context.cc
	op_utils.cc
	broadcast.cc
	convolution.cc
	cross_entropy_loss.cc
	matmul.cc
	misc.cc
	pool.cc
	relu.cc
	reshape.cc
	sigmoid.cc
	tensor_eq.cc
	tensor_binary_op.cc
	tensor_copy.cc
	tensor_reduction_op.cc
	tensor_scalar_op.cc
	tensor_unary_op.cc
	transpose.cc
)

target_link_libraries(neuroframe_op PUBLIC
	neuroframe_tensor
	neuroframe_backend_cpu
	neuroframe_backend_cuda
	neuroframe_cgraph
)