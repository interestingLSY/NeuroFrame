add_library(neuroframe_op STATIC
	op.cc
	op_context.cc
	op_utils.cc
	relu.cc
	sigmoid.cc
	tensor_eq.cc
)

target_link_libraries(neuroframe_op PUBLIC
	neuroframe_tensor
	neuroframe_backend_cpu
	neuroframe_backend_cuda)