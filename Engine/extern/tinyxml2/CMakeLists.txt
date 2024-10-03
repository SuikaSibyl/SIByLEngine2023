# CPP library 
set(PROJECT_NAME tinyxml)
add_library(${PROJECT_NAME} STATIC)
target_include_directories(${PROJECT_NAME} PUBLIC include
  "${PROJECT_SOURCE_DIR}/Engine/extern/tinyxml2/includes")
target_sources(${PROJECT_NAME} PRIVATE 
  "includes/tinyxml2.cpp")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20)