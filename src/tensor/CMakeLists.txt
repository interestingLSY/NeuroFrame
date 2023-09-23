add_library(neuroframe_tensor STATIC
	tensor.cc
)

target_link_libraries(neuroframe_tensor PUBLIC neuroframe_basic CUDA::cudart)