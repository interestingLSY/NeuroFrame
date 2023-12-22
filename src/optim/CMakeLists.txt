add_library(neuroframe_optim STATIC
	adam.cc
	optim.cc
	sgd.cc
)

target_link_libraries(neuroframe_optim PUBLIC neuroframe_basic neuroframe_tensor CUDA::cudart)
