add_executable(unittest_op
	activations.cc
	binary_ops.cc
	cross_entropy_loss.cc
	matmul.cc
	pool.cc
	scalar_ops.cc
	unary_ops.cc
	transpose.cc
)
target_link_libraries(unittest_op PUBLIC
	neuroframe_op
	gtest_main
)