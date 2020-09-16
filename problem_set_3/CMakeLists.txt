cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
 
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( /usr/local/cuda/include )


add_library(utils STATIC
  compare.cpp
  compare.h
  reference_calc.cpp
  reference_calc.h
  HW3.cu
  timer.h
  utils.h
  student_func.cu
  loadSaveImage.h
  loadSaveImage.cpp
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

set_property(TARGET utils  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(app main.cpp)

set_property(TARGET app  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(utils ${OpenCV_LIBS})
target_link_libraries(app utils ${OpenCV_LIBS})