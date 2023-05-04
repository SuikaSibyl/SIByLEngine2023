
cmake_minimum_required(VERSION 3.23)
#set(DCMAKE_TOOLCHAIN_FILE, P://vcpkg//vcpkg//scripts//buildsystems//vcpkg.cmake)
set(VCPKG_ROOT "${CMAKE_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE PATH "")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT})

set(PROJECT_NAME Core)
set(CMAKE_CXX_STANDARD 20)

add_compile_definitions(_NEED_LOG)

#set(yaml-cpp_DIR P:/vcpkg/vcpkg/installed/x64-windows/share/yaml-cpp)
#find_package(yaml-cpp CONFIG REQUIRED)

################################################################################
# Source groups
################################################################################
set(MOD_LIST
        # SE.Core.System Module
        "Public/System/SE.Core.System-Layer.ixx"
        "Public/System/SE.Core.System-Manager.ixx"
        "Public/System/SE.Core.System.ixx"
        # SE.Core.Memory Module
        "Public/Memory/SE.Core.Memory-Allocator.ixx"
        "Public/Memory/SE.Core.Memory-BlockedArray.ixx"
        "Public/Memory/SE.Core.Memory-Buffer.ixx"
        "Public/Memory/SE.Core.Memory-DoubleEndedStackAllocator.ixx"
        "Public/Memory/SE.Core.Memory-Memory.ixx"
        "Public/Memory/SE.Core.Memory-MemoryArena.ixx"
        "Public/Memory/SE.Core.Memory-MemoryManager.ixx"
        "Public/Memory/SE.Core.Memory-SmartPtr.ixx"
        "Public/Memory/SE.Core.Memory-StackAllocator.ixx"
        "Public/Memory/SE.Core.Memory.ixx"
        "Private/Memory/SE.Core.Memory-Allocator.cpp"
        "Private/Memory/SE.Core.Memory-Buffer.cpp"
        "Private/Memory/SE.Core.Memory-MemoryArena.cpp"
        "Private/Memory/SE.Core.Memory-MemoryManager.cpp"
        "Private/Memory/SE.Core.Memory-StackAllocator.cpp"
        # SE.Core.Log Module
        "Public/Print/SE.Core.Log-LogStream.ixx"
        "Public/Print/SE.Core.Log-Logger.ixx"
        "Public/Print/SE.Core.Log-LogManager.ixx"
        "Public/Print/SE.Core.Log.ixx"
        # SE.Core.ECS Module
        "Public/ECS/SE.Core.ECS-ECS.ixx"
        "Public/ECS/SE.Core.ECS-SparseSet.ixx"
        "Public/ECS/SE.Core.ECS.ixx"
        # SE.Core.Event Module
        "Public/Event/SE.Core.Event-Signal.ixx"
        "Public/Event/SE.Core.Event-Event.ixx"
        "Public/Event/SE.Core.Event.ixx"
        # SE.Core.IO Module
        "Public/IO/SE.Core.IO-SyncRW.ixx"
        "Public/IO/SE.Core.IO.ixx"
        # SE.Core.Misc Module
        "Public/Misc/SE.Core.Misc-Timer.ixx"
        "Public/Misc/SE.Core.Misc.ixx"
        # SE.Core.Resource Module
        "Public/Resource/SE.Core.Resource.ixx"
        "Public/Resource/SE.Core.Resource.cpp"
        # SE.Core.UnitTest Module
        "Public/UnitTest/SE.Core.UnitTest.ixx"
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

target_link_libraries(${PROJECT_NAME} yaml)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Core)
include_directories(${CMAKE_SOURCE_DIR}/Engine/Source/ThirdParty/yaml/include)