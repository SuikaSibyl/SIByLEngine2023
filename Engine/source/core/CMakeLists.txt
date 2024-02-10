# CPP library 
set(PROJECT_NAME core)
add_library(${PROJECT_NAME} SHARED)
target_include_directories(${PROJECT_NAME} PUBLIC include
  "${PROJECT_SOURCE_DIR}/Engine/extern/glfw/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/aftermath/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vma/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/spirv_reflect/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/slang/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vulkan/include")
target_link_directories(${PROJECT_NAME} PUBLIC
  "${PROJECT_SOURCE_DIR}/Engine/extern/glfw/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vulkan/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/slang/libs")
target_link_libraries(${PROJECT_NAME} PRIVATE 
  ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} 
  "glfw3.lib" "vulkan-1.lib" "slang.lib" "slang-rt.lib")
target_sources(${PROJECT_NAME} PRIVATE 
  "source/base/se.log.cpp"
  "source/base/se.memory.cpp"
  "source/base/se.platform.win.cpp"
  "source/base/se.window.gltf.cpp"
  "source/rhi/se.vma.cpp"
  "source/rhi/se.rhi.impl.cpp"
  "source/rhi/se.rhi.vk.cpp"
  "source/rhi/se.rhi.cu.cpp"
  "source/rhi/se.rhi.torch.cpp"
  "source/gfx/ex.spirv_reflect.cpp"
  "source/gfx/se.gfx.cpp"
  "source/base/se.resource.cpp"
  "source/rdg/se.rdg.cpp")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20)

# PY library 
set(PROJECT_NAME pycore)
add_library(${PROJECT_NAME} SHARED)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PRIVATE core ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_sources(${PROJECT_NAME} PRIVATE "pyscript/py.core.cpp")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20
    SUFFIX ".pyd")
# target_compile_options(pytorch_cmake_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin g++-9>)