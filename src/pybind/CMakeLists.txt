set(PYMODULE_NAME "neuroframe")
# Copied from https://github.com/pybind/cmake_example
pybind11_add_module(${PYMODULE_NAME} MODULE
	basic.cc
	cgraph.cc
	ops.cc
	pybind.cc
	tensor.cc
)

target_link_libraries(${PYMODULE_NAME} PRIVATE
	neuroframe_backend_cpu
	neuroframe_backend_cuda
	neuroframe_basic
	neuroframe_cgraph
	neuroframe_op
	neuroframe_tensor
)

target_compile_options(${PYMODULE_NAME} PRIVATE
	-flto=auto	# Supress the warning "lto-wrapper: warning: using serial compilation of 2 LTRANS jobs"
)

# Use stubgen to create .pyi files to sit alongside the just-built python module
# Copied from https://stackoverflow.com/a/77456943/16569836
set(Stubgen_Executable "/usr/bin/stubgen")
add_custom_command(TARGET ${PYMODULE_NAME} POST_BUILD
    COMMAND ${Stubgen_Executable} -p ${PYMODULE_NAME} -o .
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Use stubgen to create .pyi for statement completion"
)