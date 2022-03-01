cmake_minimum_required(VERSION 3.10)

project(RegionGame LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52;70;75)
set(CMAKE_BUILD_TYPE Release)

add_executable(main main.cu)

target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)