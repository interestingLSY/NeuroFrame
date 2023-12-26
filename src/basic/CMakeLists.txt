add_library(neuroframe_basic STATIC
	device.cc
	inference_mode.cc
	log.cc
	mem.cc
	mem_pool/abstract_mem_pool.cc
	mem_pool/cpu_mem_pool.cc
	mem_pool/cuda_mem_pool.cc
	mem_pool/mem_pool.cc
	random.cc
	scalar.cc
)

target_link_libraries(neuroframe_basic PUBLIC CUDA::cudart)
