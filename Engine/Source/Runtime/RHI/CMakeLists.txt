cmake_minimum_required(VERSION 3.23)
# Set project setting
set(PROJECT_NAME RHI)
set(CMAKE_CXX_STANDARD 20)
# Add target library
file(GLOB_RECURSE SRC_FILES "Private/*.cpp")
add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
        PRIVATE
        ${SRC_FILES})
# Set c++ standard 20
# Set In Folder - Dependency
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CXX_STANDARD 20
        USE_FOLDERS ON
        FOLDER Graphics)
# Set Include Folder
target_include_directories(${PROJECT_NAME}
        PUBLIC
        Public
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/vma/include
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/glad/include
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/glfw/include
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/vulkan/include)
target_link_libraries(${PROJECT_NAME}
        Prelude
        Core
        Math
        Platform)