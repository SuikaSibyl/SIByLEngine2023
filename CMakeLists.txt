cmake_minimum_required(VERSION 3.8)

project("SIByLEngine2023")

add_subdirectory("Engine/Source/ThirdParty/yaml")
add_subdirectory("Engine/Source/ThirdParty/imgui")

add_subdirectory("Engine/Source/Runtime/Utility")
add_subdirectory("Engine/Source/Runtime/Core")
add_subdirectory("Engine/Source/Runtime/Math")
add_subdirectory("Engine/Source/Runtime/Platform")
add_subdirectory("Engine/Source/Runtime/Parallelism")

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
