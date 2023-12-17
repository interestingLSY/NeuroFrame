add_library(neuroframe_basic STATIC
	device.cc
	inference_mode.cc
	log.cc
	mem.cc
	random.cc
	scalar.cc
)

target_link_libraries(neuroframe_basic PUBLIC CUDA::cudart)
