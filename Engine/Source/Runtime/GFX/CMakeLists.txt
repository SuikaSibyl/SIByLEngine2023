cmake_minimum_required(VERSION 3.23)
# Set project setting
set(PROJECT_NAME GFX)
set(CMAKE_CXX_STANDARD 20)
# Add target library
file(GLOB_RECURSE SRC_FILES "Public/*.hpp" "Private/*.cpp")
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
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/assimp
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/tinyobjloader
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/tinygltf/include
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/yaml/include
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/spirvcross/include
        ${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/openfbx/include)
target_link_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/assimp
        ${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/glfw
        ${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/vulkan
        ${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/spirvcross
        )
target_link_libraries(${PROJECT_NAME} PUBLIC Prelude Core RHI Image yaml
        PUBLIC
        assimp-vc143-mt.lib
        glfw3.lib
        vulkan-1.lib
        opengl32.lib)
if (CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    target_link_libraries(${PROJECT_NAME} PUBLIC
            spirv-cross-cored.lib
            spirv-cross-cppd.lib
            spirv-cross-glsld.lib)
else ()
    target_link_libraries(${PROJECT_NAME} PUBLIC
            spirv-cross-core.lib
            spirv-cross-cpp.lib
            spirv-cross-glsl.lib)
endif ()
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/assimp/assimp-vc143-mt.dll"
        "${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE}/assimp-vc143-mt.dll")