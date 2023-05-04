cmake_minimum_required(VERSION 3.23)

set(PROJECT_NAME imgui)
set(CMAKE_CXX_STANDARD 20)


################################################################################
# Source groups
################################################################################

file(GLOB_RECURSE INC_FILES "includes/*.h")
file(GLOB_RECURSE SRC_FILES "includes/*.cpp")

set(ALL_FILES
        ${INC_FILES}
        ${SRC_FILES}
        )

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_library(${PROJECT_NAME})
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${ALL_FILES}
            )

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    add_library(${PROJECT_NAME})
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${ALL_FILES}
            )

    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

endif ()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Dependency)
target_include_directories(${PROJECT_NAME} PRIVATE includes)

include_directories(${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/glfw/include)
include_directories(${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/vulkan/include)