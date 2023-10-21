add_library(neuroframe_backend_cpu STATIC
	relu.cc
	sigmoid.cc
	tensor_eq.cc
	transpose.cc
)

target_link_libraries(neuroframe_backend_cpu PUBLIC neuroframe_tensor)