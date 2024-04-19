# CPP library 
set(PROJECT_NAME editor)
add_compile_options(-msse2 -msse3)
add_library(${PROJECT_NAME} SHARED)
target_include_directories(${PROJECT_NAME} PUBLIC include
  "${PROJECT_SOURCE_DIR}/Engine/extern/glfw/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vma/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/spirv_reflect/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/slang/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/tinygltf/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vulkan/include")
target_link_directories(${PROJECT_NAME} PUBLIC
  "${PROJECT_SOURCE_DIR}/Engine/extern/glfw/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vulkan/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/slang/libs")
target_link_libraries(${PROJECT_NAME} PUBLIC imgui core)
target_sources(${PROJECT_NAME} PRIVATE 
  "source/seditor-base.cpp"
  "source/seditor-gfx.cpp"
  "source/seditor-rdg.cpp")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20)
    
# PY library 
set(PROJECT_NAME pyeditor)
add_library(${PROJECT_NAME} SHARED)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PRIVATE core editor ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_sources(${PROJECT_NAME} PRIVATE "pyscript/py.editor.cpp")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20
    SUFFIX ".pyd")