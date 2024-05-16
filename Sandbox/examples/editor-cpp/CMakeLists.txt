set(PROJECT_NAME editor-cpp)
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE
  "main.cpp" "source/geometry_inspector.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC core editor ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/se")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/Engine/binary/dlls/${CURRENT_PLATFORM}"
    "${CMAKE_CURRENT_SOURCE_DIR}/se")