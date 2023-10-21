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
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -O1 -fsanitize=address")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -DDEBUG -arch=native")
elseif (CMAKE_BUILD_TYPE MATCHES PROFILE)
	message("Building in PROFILE mode")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -DNDEBUG -pg")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --use_fast_math -arch=native")
else()
	message("Building in RELEASE mode")
	# NOTE. "assert" won't be evaluated when NDEBUG is defined
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -DNDEBUG")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --use_fast_math -arch=native")
endif()

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
