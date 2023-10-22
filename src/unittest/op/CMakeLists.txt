add_executable(unittest_op
	activations.cc
	binary_ops.cc
	transpose.cc
	matmul.cc
	unary_ops.cc
	pool.cc
	cross_entropy_loss.cc
)
target_link_libraries(unittest_op PUBLIC
	neuroframe_op
	gtest_main
)