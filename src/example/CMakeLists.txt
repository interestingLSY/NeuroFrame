add_executable(example_tensor tensor.cc)
target_link_libraries(example_tensor neuroframe_tensor)

add_executable(example_op op.cc)
target_link_libraries(example_op neuroframe_tensor neuroframe_op)

add_executable(hw1 hw1.cc)
target_link_libraries(hw1 neuroframe_tensor neuroframe_op)

add_executable(hw2 hw2.cc)
target_link_libraries(hw2 neuroframe_tensor neuroframe_op)
