add_executable(unittest_op
	activations.cc
	transpose.cc
	matmul.cc
)
target_link_libraries(unittest_op PUBLIC
	neuroframe_op
	gtest_main
)