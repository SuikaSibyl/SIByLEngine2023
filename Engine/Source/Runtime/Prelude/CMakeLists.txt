cmake_minimum_required(VERSION 3.23)
# Set project setting
set(PROJECT_NAME Prelude)
set(CMAKE_CXX_STANDARD 20)
# Add target library
file(GLOB_RECURSE SRC_FILES "Public/*.hpp")
add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
        PRIVATE
        ${SRC_FILES})
# Set c++ standard 20
# Set In Folder - Dependency
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 20
        USE_FOLDERS ON
        FOLDER Core)
# Set Include Folder
target_include_directories(${PROJECT_NAME} PUBLIC Public)