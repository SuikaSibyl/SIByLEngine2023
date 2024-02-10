cmake_minimum_required (VERSION 3.22)

project(SIByL LANGUAGES CXX CUDA)

include_directories("S:\\Anaconda\\envs\\torch21\\lib\\site-packages\\pybind11\\include")
include_directories("S:\\Anaconda\\envs\\torch21\\include")
link_directories("S:\\Anaconda\\envs\\torch21\\libs")

set(CMAKE_PREFIX_PATH "S:\\Anaconda\\envs\\torch21\\lib\\site-packages\\torch\\share\\cmake")
find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "S:\\Anaconda\\envs\\torch21\\Lib\\site-packages\\torch\\lib")
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory("Engine")
add_subdirectory("Sandbox")