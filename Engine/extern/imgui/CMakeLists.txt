# CPP library 
set(PROJECT_NAME imgui)
add_library(${PROJECT_NAME} STATIC)
target_include_directories(${PROJECT_NAME} PUBLIC include
  "${PROJECT_SOURCE_DIR}/Engine/extern/imgui/includes"
  "${PROJECT_SOURCE_DIR}/Engine/extern/glfw/include/glfw"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vulkan/include")
target_link_directories(${PROJECT_NAME} PUBLIC
  "${PROJECT_SOURCE_DIR}/Engine/extern/glfw/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vulkan/libs")
target_sources(${PROJECT_NAME} PRIVATE 
  "includes/imgui.cpp"
  "includes/imgui_demo.cpp"
  "includes/imgui_draw.cpp"
  "includes/imgui_tables.cpp"
  "includes/imgui_widgets.cpp"
  "includes/backends/imgui_impl_glfw.cpp"
  "includes/backends/imgui_impl_vulkan.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} 
  "glfw3.lib" "vulkan-1.lib")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20)