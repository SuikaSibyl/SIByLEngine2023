# CPP library 
set(PROJECT_NAME cnpy)
add_library(${PROJECT_NAME} STATIC)
target_include_directories(${PROJECT_NAME} PUBLIC include
  "${PROJECT_SOURCE_DIR}/Engine/extern/cnpy/includes")
target_link_libraries(${PROJECT_NAME} PUBLIC 
  "${PROJECT_SOURCE_DIR}/Engine/extern/zlib/zlib.lib"
  "${PROJECT_SOURCE_DIR}/Engine/extern/zlib/zlibstatic.lib")
target_sources(${PROJECT_NAME} PRIVATE 
  "includes/cnpy.cpp")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20)