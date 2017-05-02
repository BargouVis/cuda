cmake_minimum_required(VERSION 2.8)
project( ProgCuda )
find_package( OpenCV REQUIRED )
find_package(CUDA QUIET REQUIRED)
file(GLOB SOURCES *.cu *.cpp)
include_directories( ${OpenCV_INCLUDE_DIRS})
cuda_add_executable(ProgCuda ${SOURCES} )
#link_directories(
#/usr/local/cuda/lib64
#/usr/local/lib
#)
#target_link_libraries(ProgCuda ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES(ProgCuda ${OpenCV_LIBS} ${CUDA_LIBRARIES} /usr/local/cuda/lib64/libcudart.so)
