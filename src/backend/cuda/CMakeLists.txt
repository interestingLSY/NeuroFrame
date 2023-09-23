add_library(neuroframe_backend_cuda STATIC
	relu.cu
	sigmoid.cu
)

target_link_libraries(neuroframe_backend_cuda PUBLIC neuroframe_tensor CUDA::cudart)