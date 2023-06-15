cmake_minimum_required(VERSION 3.8)

project("SIByLEngine2023")

set(CURRENT_PLATFORM "x64")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Engine/Binaries/Source/${CURRENT_PLATFORM})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Engine/Binaries/Source/${CURRENT_PLATFORM})

add_compile_definitions(_NEED_LOG)

add_subdirectory("Engine/Source/ThirdParty/yaml")
add_subdirectory("Engine/Source/ThirdParty/imgui")

add_subdirectory("Engine/Source/Runtime/Prelude")
add_subdirectory("Engine/Source/Runtime/Core")
add_subdirectory("Engine/Source/Runtime/Math")
add_subdirectory("Engine/Source/Runtime/Platform")
#add_subdirectory("Engine/Source/Runtime/Parallelism")

add_subdirectory("Engine/Source/Runtime/HSL")
add_subdirectory("Engine/Source/Runtime/RHI")
add_subdirectory("Engine/Source/Runtime/Image")
add_subdirectory("Engine/Source/Runtime/GFX")
add_subdirectory("Engine/Source/Runtime/Video")

add_subdirectory("Engine/Source/Editor/EditorCore")
add_subdirectory("Engine/Source/Editor/EditorGFX")
add_subdirectory("Engine/Source/Editor/EditorRDG")
add_subdirectory("Engine/Source/Editor/Editor")

add_subdirectory("Engine/Source/Runtime/RDG")
add_subdirectory("Engine/Source/Runtime/SRenderer")
add_subdirectory("Engine/Source/Runtime/Application")

add_subdirectory("Sandbox")
add_subdirectory("APITest")
