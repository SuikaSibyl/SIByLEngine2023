cmake_minimum_required(VERSION 3.23)
# Set project setting
set(PROJECT_NAME Sandbox)
set(CMAKE_CXX_STANDARD 20)
# Add target library
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
        PRIVATE
        ${SRC_FILES})
# Set c++ standard 20
# Set In Folder - Dependency
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CXX_STANDARD 20)
# Set Include Folder
target_include_directories(${PROJECT_NAME}
        PUBLIC
        Public
        ${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/glfw
        ${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/spirvcross
        ${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/vulkan)
target_link_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/ffmpeg)
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/Sandbox")


target_link_libraries(${PROJECT_NAME} PUBLIC
        Prelude Core Platform Math RHI GFX RDG
        Image Video imgui SRenderer EditorCore EditorGFX EditorRDG Editor Application)
