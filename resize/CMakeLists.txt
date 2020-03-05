# OCRTranformer TensorRT implementation

cmake_minimum_required(VERSION 3.12.1 FATAL_ERROR)

project(OCRTrt
        LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Found TensorRT libs at ${TENSORRT_LIBRARY}")

find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR "Cannot find TensorRT library.")
endif()
# 添加头文件
# set( COMMON ${CMAKE_SOURCE_DIR}/../common)
# if( IS_DIRECTORY ${COMMON} )
#   add_subdirectory(${COMMON} common.out)
# else()
#   message(FATAL_ERROR   "INVALID FOLDER 'COMMON'=${COMMON}" )
# endif()

include_directories(common)
add_subdirectory(common)
#Add the source in project root directory
AUX_SOURCE_DIRECTORY(. DIRSRCS)
add_executable(OCRTrt resizeNearestPlugin.h resizeNearestPlugin.cpp resize.cu plugin.h kernel.h ${COMMON} test1.cpp )
# add_library(OCRTrt SHARED resizeNearestPlugin.h resizeNearestPlugin.cpp resize.cu plugin.h kernel.h )

TARGET_LINK_LIBRARIES(OCRTrt ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${COMMON})
