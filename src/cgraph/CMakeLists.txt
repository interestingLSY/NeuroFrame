add_library(neuroframe_cgraph STATIC
	cgraph_edge.cc
	cgraph_node.cc
	cgraph.cc
)

target_link_libraries(neuroframe_cgraph PUBLIC CUDA::cudart)
