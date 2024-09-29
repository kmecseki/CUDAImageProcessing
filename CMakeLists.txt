cmake_minimum_required(VERSION 3.25.1) # Debian bookworm current 

project(CUDAImageProcessing)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_89 -allow-unsupported-compiler") # RTX 4080

include_directories(${OpenCV_INCLUDE_DIRS} include)

cuda_add_library(cudakernels src/cuda_kernels.cu)
add_executable(main src/main.cpp src/utils.cpp)
target_link_libraries(main cudakernels ${OpenCV_LIBS} cuda)

