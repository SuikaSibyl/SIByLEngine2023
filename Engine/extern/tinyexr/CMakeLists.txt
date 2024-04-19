# CPP library 
set(PROJECT_NAME tinyexr)
add_library(${PROJECT_NAME} STATIC)
target_include_directories(${PROJECT_NAME} PUBLIC include
  "${PROJECT_SOURCE_DIR}/Engine/extern/tinyexr/includes")
target_sources(${PROJECT_NAME} PRIVATE 
  "includes/miniz.cpp"
  "includes/tinyexr.cpp" )
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20)