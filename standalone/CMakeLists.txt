cmake_minimum_required(VERSION 3.21)
project(gpu_processor)

find_package(hip REQUIRED)

add_executable(gpu_processor src/main.cpp)
target_link_libraries(gpu_processor PRIVATE hip::device)