cmake_minimum_required(VERSION 3.23)

set(PROJECT_NAME Sandbox)
set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE SRC_FILES "src/main.cpp")

################################################################################
# Source groups
################################################################################
set(MOD_LIST ${MOD_FILES})
set(SRC_LIST ${SRC_FILES})

set(ALL_FILES
        ${MOD_LIST}
        ${SRC_LIST}
        )

link_directories(${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/glfw)
link_directories(${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/spirvcross)
link_directories(${CMAKE_SOURCE_DIR}/Engine/Binaries/ThirdParty/vulkan)

link_libraries(glfw3.lib)
link_libraries(opengl32.lib)
link_libraries(vulkan-1.lib)
if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    link_libraries(spirv-cross-cored.lib)
    link_libraries(spirv-cross-cppd.lib)
    link_libraries(spirv-cross-glsld.lib)
    message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")
else()
    link_libraries(spirv-cross-core.lib)
    link_libraries(spirv-cross-cpp.lib)
    link_libraries(spirv-cross-glsl.lib)
    message("Release mode:${CMAKE_C_FLAGS_RELEASE}")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
    cmake_minimum_required(VERSION 3.26)
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API 2182bf5c-ef0d-489a-91da-49dbc3090d2a)
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")

    add_executable(${PROJECT_NAME})
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${SRC_LIST}
            PRIVATE FILE_SET CXX_MODULES FILES
            ${MOD_LIST}
            )

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    add_executable(${PROJECT_NAME})
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${SRC_LIST}
            ${MOD_LIST}
            )

    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

endif ()

target_link_libraries(${PROJECT_NAME} Core)
target_link_libraries(${PROJECT_NAME} Utility)
target_link_libraries(${PROJECT_NAME} Platform)
target_link_libraries(${PROJECT_NAME} Parallelism)
target_link_libraries(${PROJECT_NAME} Math)
target_link_libraries(${PROJECT_NAME} RHI)
target_link_libraries(${PROJECT_NAME} GFX)
target_link_libraries(${PROJECT_NAME} RDG)
target_link_libraries(${PROJECT_NAME} Image)
target_link_libraries(${PROJECT_NAME} Video)
target_link_libraries(${PROJECT_NAME} imgui)
target_link_libraries(${PROJECT_NAME} SRenderer)
target_link_libraries(${PROJECT_NAME} EditorCore)
target_link_libraries(${PROJECT_NAME} EditorGFX)
target_link_libraries(${PROJECT_NAME} Editor)
target_link_libraries(${PROJECT_NAME} Application)

include_directories(${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/glad/include)
include_directories(${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/tinygltf/include)
include_directories(${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/imgui/includes)