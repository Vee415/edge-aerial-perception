cmake_minimum_required(VERSION 3.18)
project(drone_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Find packages ----
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# ---- TensorRT paths ----
# On Jetson: headers at /usr/include/aarch64-linux-gnu/, libs at /usr/lib/aarch64-linux-gnu/
set(TRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu" CACHE PATH "TensorRT include dir")
set(TRT_LIB_DIR "/usr/lib/aarch64-linux-gnu" CACHE PATH "TensorRT lib dir")

# ---- CUDA ----
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include" CACHE PATH "CUDA include dir")
set(CUDA_LIB_DIR "/usr/local/cuda/lib64" CACHE PATH "CUDA lib dir")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")

# ---- CUDA architecture for Jetson Orin Nano Super ----
# sm_87 = Ampere (Orin). Adjust for other GPUs.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87")

# ---- Compiler warnings ----
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# ---- Sources ----
set(SOURCES
    src/main.cpp
    src/trt_engine.cpp
    src/cuda_preprocess.cu
    src/nms.cpp
    src/video_io.cpp
)

add_executable(drone_inference ${SOURCES})

target_include_directories(drone_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Link TensorRT, CUDA runtime, and OpenCV
link_directories(${TRT_LIB_DIR} ${CUDA_LIB_DIR})

target_link_libraries(drone_inference
    nvinfer
    cudart
    ${OpenCV_LIBS}
)

# ---- Install ----
install(TARGETS drone_inference DESTINATION bin)