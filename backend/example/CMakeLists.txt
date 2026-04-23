# Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.22)

project(tutorialrecommendedbackend LANGUAGES C CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../standalone/src)

#
# Options
#
# GPU support options for both NVIDIA and AMD
#
option(TRITON_ENABLE_CUDA "Enable NVIDIA CUDA support in backend" OFF)
option(TRITON_ENABLE_ROCM "Enable AMD ROCm support" OFF)
option(TRITON_ENABLE_ALPAKA "Use Alpaka portability layer (requires TRITON_ENABLE_ROCM)" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/backend repo")

# ROCm path configuration
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")

#
# Setting C++ min standard
#
set(TRITON_MIN_CXX_STANDARD 20 CACHE STRING "The minimum C++ standard whose features are requested to build this target.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# ROCm/HIP Configuration
#
if(TRITON_ENABLE_ROCM)
  set(TRITON_ROCM_HOME ${ROCM_PATH})

  set(ENV{ROCM_PATH} ${TRITON_ROCM_HOME})
  set(ENV{HIP_PATH} ${TRITON_ROCM_HOME})

  list(APPEND CMAKE_PREFIX_PATH
    ${TRITON_ROCM_HOME}
    ${TRITON_ROCM_HOME}/hip
    ${TRITON_ROCM_HOME}/llvm
    ${TRITON_ROCM_HOME}/lib/cmake
    $ENV{ROCM_PATH}
    $ENV{HIP_PATH}
  )

  file(GLOB rocm_cmake_components ${TRITON_ROCM_HOME}/lib/cmake/*)
  list(APPEND CMAKE_PREFIX_PATH ${rocm_cmake_components})

  if(NOT DEFINED CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER "${TRITON_ROCM_HOME}/llvm/bin/clang++")
  endif()

  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx940;gfx941;gfx942" CACHE STRING "HIP architectures")
  endif()

  enable_language(HIP)
  find_package(hip REQUIRED CONFIG PATHS ${TRITON_ROCM_HOME})

  include_directories(
    ${TRITON_ROCM_HOME}/include
    ${TRITON_ROCM_HOME}/hip/include
    ${HIP_INCLUDE_DIRS}
  )
  link_directories(
    ${TRITON_ROCM_HOME}/lib
    ${TRITON_ROCM_HOME}/hip/lib
  )

  add_compile_definitions(TRITON_ENABLE_ROCM __HIP_PLATFORM_AMD__)
  set(HIP_PLATFORM "amd" CACHE STRING "HIP platform")

  if(EXISTS "${TRITON_ROCM_HOME}/.info/version-dev")
    file(READ "${TRITON_ROCM_HOME}/.info/version-dev" ROCM_VERSION_DEV_RAW)
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-.*$" ROCM_VERSION_DEV_MATCH ${ROCM_VERSION_DEV_RAW})
    if(ROCM_VERSION_DEV_MATCH)
      set(ROCM_VERSION_DEV_MAJOR ${CMAKE_MATCH_1})
      set(ROCM_VERSION_DEV_MINOR ${CMAKE_MATCH_2})
      set(ROCM_VERSION_DEV_PATCH ${CMAKE_MATCH_3})
      set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
      math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")
    endif()
  endif()

  message(STATUS "ROCm support enabled")
  message(STATUS "ROCm path: ${TRITON_ROCM_HOME}")
  message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
  message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
  if(DEFINED ROCM_VERSION_DEV)
    message(STATUS "ROCm version: ${ROCM_VERSION_DEV} (${ROCM_VERSION_DEV_INT})")
  endif()
endif()

#
# Alpaka Configuration
#
if(TRITON_ENABLE_ALPAKA)
  if(NOT TRITON_ENABLE_ROCM)
    message(FATAL_ERROR "TRITON_ENABLE_ALPAKA requires TRITON_ENABLE_ROCM")
  endif()
  find_package(alpaka REQUIRED HINTS /usr/local/lib/cmake/alpaka)
  add_compile_definitions(TRITON_ENABLE_ALPAKA ALPAKA_ACC_GPU_HIP_ENABLED)
  message(STATUS "Alpaka portability layer enabled (HIP backend)")
endif()

#
# CUDA Configuration (for NVIDIA GPUs)
#
if(TRITON_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  add_compile_definitions(TRITON_ENABLE_CUDA)
  message(STATUS "CUDA support enabled")
endif()

#
# Dependencies
#

# This must be set before FetchContent to propagate to all subprojects
set(BUILD_TESTING OFF CACHE BOOL "Disable all testing" FORCE)
set(TRITON_ENABLE_TESTS OFF CACHE BOOL "Disable Triton tests" FORCE)
set(TRITON_ENABLE_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
# repo-backend enables CUDA unconditionally when TRITON_ENABLE_GPU=ON (its default).
# Only keep it on if we are also building our CUDA backend.
if(NOT TRITON_ENABLE_CUDA)
  set(TRITON_ENABLE_GPU OFF CACHE BOOL "Disable GPU support in repo-backend" FORCE)
endif()

include(FetchContent)

# Fetch from Git (original behavior)
FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)

FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_recommended.ldscript libtriton_recommended.ldscript COPYONLY)

add_library(
  triton-recommended-backend SHARED
  src/recommended.cc
)

# Force HIP compiler (clang++) for .cc files when ROCm is enabled,
# since CMake doesn't recognize .cc as a HIP source by default.
# Also explicitly propagate Triton include dirs to the HIP compiler,
# as CMake does not always forward INTERFACE_INCLUDE_DIRECTORIES from
# CXX targets to HIP-language source files.
if(TRITON_ENABLE_ROCM)
  set_source_files_properties(
    src/recommended.cc
    PROPERTIES LANGUAGE HIP
  )
  target_include_directories(
    triton-recommended-backend
    PRIVATE
      $<TARGET_PROPERTY:triton-backend-utils,INTERFACE_INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:triton-core-backendapi,INTERFACE_INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:triton-core-serverapi,INTERFACE_INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:triton-core-serverstub,INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_include_directories(triton-recommended-backend PRIVATE
    ${TRITON_ROCM_HOME}/hip/include
  )
elseif(TRITON_ENABLE_CUDA)
  set_source_files_properties(
    src/recommended.cc
    PROPERTIES LANGUAGE CUDA
  )
endif()

add_library(
  TutorialRecommendedBackend::triton-recommended-backend ALIAS triton-recommended-backend
)

target_include_directories(
  triton-recommended-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-recommended-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-recommended-backend PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX,HIP>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

# Base libraries
target_link_libraries(
  triton-recommended-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

# Add ROCm/HIP libraries if enabled
if(TRITON_ENABLE_ROCM)
  target_link_libraries(
    triton-recommended-backend
    PRIVATE
      hip::host
      hip::device
  )
  
  # Add ROCm runtime libraries
  target_link_libraries(
    triton-recommended-backend
    PRIVATE
      ${ROCM_PATH}/lib/libamdhip64.so
  )
  
  # Optional: Add rocBLAS, rocRAND, etc. if needed
  # target_link_libraries(triton-recommended-backend PRIVATE roc::rocblas)

  if(TRITON_ENABLE_ALPAKA)
    target_link_libraries(triton-recommended-backend PRIVATE alpaka::alpaka)
    target_include_directories(triton-recommended-backend PRIVATE
      $<TARGET_PROPERTY:alpaka::alpaka,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
endif()

# Add CUDA libraries if enabled
if(TRITON_ENABLE_CUDA)
  target_link_libraries(
    triton-recommended-backend
    PRIVATE
      CUDA::cudart
  )
endif()

if(WIN32)
  set_target_properties(
    triton-recommended-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_recommended
  )
else()
  set_target_properties(
    triton-recommended-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_recommended
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_recommended.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_recommended.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TutorialRecommendedBackend)

install(
  TARGETS
    triton-recommended-backend
  EXPORT
    triton-recommended-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/../../models/example
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/../../models/example
)

install(
  EXPORT
    triton-recommended-backend-targets
  FILE
    TutorialRecommendedBackendTargets.cmake
  NAMESPACE
    TutorialRecommendedBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TutorialRecommendedBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TutorialRecommendedBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TutorialRecommendedBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-recommended-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TutorialRecommendedBackendTargets.cmake
  NAMESPACE TutorialRecommendedBackend::
)

export(PACKAGE TutorialRecommendedBackend)