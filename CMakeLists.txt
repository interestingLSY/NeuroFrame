cmake_minimum_required (VERSION 3.10.0)
project(
	"NeuroFrame"
	VERSION 1.0.0
	LANGUAGES CXX CUDA
)

find_package(CUDAToolkit 11.4 REQUIRED)

# Make all executables goto bin directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
	message("Building in DEBUG mode")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -O1 -fsanitize=address -Wno-unused-parameter -fPIC")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -DDEBUG -arch=native -fPIC")
elseif (CMAKE_BUILD_TYPE MATCHES PROFILE)
	message("Building in PROFILE mode")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -DNDEBUG -pg -Wno-unused-parameter -fPIC")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --use_fast_math -arch=native -fPIC")
else()
	message("Building in RELEASE mode")
	# NOTE. "assert" won't be evaluated when NDEBUG is defined
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -DNDEBUG -Wno-unused-parameter -fPIC")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --use_fast_math -arch=native -fPIC")
endif()

find_package(pybind11 REQUIRED)

# # Set up COMMON_HEADER_DIRS and COMMON_LIB_DIRS
# set(COMMON_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}
#   ${PROJECT_SOURCE_DIR}/src
# )
# set(COMMON_LIB_DIRS "")
# set(COMMON_LIBS CUDA::cudart)

# # Add Python into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
# set(PYTHON_PATH "python" CACHE STRING "Python path")
# execute_process(COMMAND ${PYTHON_PATH} "-c" "import sysconfig;
# print(sysconfig.get_paths()['include']);"
#                   RESULT_VARIABLE _PYTHON_SUCCESS
#                   OUTPUT_VARIABLE PY_INCLUDE_DIR)
# if (NOT _PYTHON_SUCCESS MATCHES 0)
# 	message(FATAL_ERROR "Python config Error.")
# endif()
# list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

# # Add LibTorch into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
# execute_process(COMMAND ${PYTHON_PATH} "-c" "import os; import torch;
# print(os.path.dirname(torch.__file__), end='');"
#                   RESULT_VARIABLE _PYTHON_SUCCESS
#                   OUTPUT_VARIABLE TORCH_DIR)
# if (NOT _PYTHON_SUCCESS MATCHES 0)
# 	message(FATAL_ERROR "Torch config Error.")
# endif()
# list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
# set(CAFFE2_USE_CUDNN 1)
# find_package(Torch REQUIRED)
# list(APPEND COMMON_HEADER_DIRS "${TORCH_INCLUDE_DIRS}")
# list(APPEND COMMON_LIBS "${TORCH_LIBRARIES}")

# # Let COMMON_HEADER_DIRS & COMMON_LIB_DIRS take effect
# include_directories(${COMMON_HEADER_DIRS})
# link_directories(${COMMON_LIB_DIRS})
# link_libraries(${COMMON_LIBS})

# # Should turn off CXX11 ABI if pytorch is built with CXX11 ABI off
# execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch;
# print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
#                   RESULT_VARIABLE _PYTHON_SUCCESS
#                   OUTPUT_VARIABLE USE_CXX11_ABI)
# message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")
# if (USE_CXX11_ABI)
#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
# else()
#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
# endif()

# GoogleTest Preparation - Code block copied from
#   https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://gitee.com/mirrors/googletest.git	# We use the mirror from gitee.com to avoid the GFW (the original repo is https://github.com/google/googletest.git)
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src)
