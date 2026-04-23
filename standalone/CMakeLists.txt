cmake_minimum_required(VERSION 3.21)
project(gpu_processor LANGUAGES CXX)

option(ENABLE_HIP    "Build the HIP backend example"    OFF)
option(ENABLE_ALPAKA "Build the Alpaka backend example" OFF)
option(ENABLE_CUDA   "Build the CUDA backend example"   OFF)

if(ENABLE_HIP)
    enable_language(HIP)
    find_package(hip REQUIRED)
    add_executable(gpu_processor_hip src/main.cpp)
    set_source_files_properties(src/main.cpp
        TARGET_DIRECTORY gpu_processor_hip
        PROPERTIES LANGUAGE HIP)
    target_link_libraries(gpu_processor_hip PRIVATE hip::device)
endif()

if(ENABLE_ALPAKA)
    find_package(alpaka REQUIRED)
    add_executable(gpu_processor_alpaka src/main.cpp)
    target_compile_definitions(gpu_processor_alpaka PRIVATE
        USE_ALPAKA_BACKEND
        ALPAKA_ACC_GPU_HIP_ENABLED)
    target_link_libraries(gpu_processor_alpaka PRIVATE alpaka::alpaka)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(gpu_processor_cuda src/main.cpp)
    set_source_files_properties(src/main.cpp
        TARGET_DIRECTORY gpu_processor_cuda
        PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(gpu_processor_cuda PRIVATE USE_CUDA_BACKEND)
    target_link_libraries(gpu_processor_cuda PRIVATE CUDA::cudart)
endif()
