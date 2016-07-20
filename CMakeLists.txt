cmake_minimum_required(VERSION 3.5)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(PROJECT ProcCL)
project(${PROJECT})

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR} src)

set(PROJECT_SRCS src/main.cpp src/Processor.cpp)
set(CMAKE_CXX_STANDARD 11)

add_executable(${PROJECT} ${PROJECT_SRCS})
target_link_libraries(${PROJECT} ${OpenCL_LIBRARY})
