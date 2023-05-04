cmake_minimum_required(VERSION 3.23)

set(PROJECT_NAME Video)
set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE MOD_FILES "Public/*.ixx")

################################################################################
# Source groups
################################################################################
set(MOD_LIST ${MOD_FILES}
        )
set(SRC_LIST
        )

set(ALL_FILES
        ${MOD_LIST}
        ${SRC_LIST}
        )

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
    cmake_minimum_required(VERSION 3.26)
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API 2182bf5c-ef0d-489a-91da-49dbc3090d2a)
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")

    add_library(${PROJECT_NAME})
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${SRC_LIST}
            PRIVATE FILE_SET CXX_MODULES FILES
            ${MOD_LIST}
            )

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    add_library(${PROJECT_NAME})
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${SRC_LIST}
            ${MOD_LIST}
            )

    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

endif ()

target_link_libraries(${PROJECT_NAME} Core)
target_link_libraries(${PROJECT_NAME} RHI)
target_link_libraries(${PROJECT_NAME} GFX)
target_link_libraries(${PROJECT_NAME} Image)
target_link_libraries(${PROJECT_NAME} yaml)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Graphics)

include_directories(${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/vulkan/Include)
include_directories(${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/yaml/include)