# Set project setting
set(PROJECT_NAME yaml)
set(CMAKE_CXX_STANDARD 20)
# Add target library
file(GLOB_RECURSE INC_FILES "include/*.h")
file(GLOB_RECURSE SRC_FILES "src/*.h" "src/*.cpp")
add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
        PRIVATE
        ${INC_FILES}
        ${SRC_FILES})
# Set c++ standard 20
# Set In Folder - Dependency
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CXX_STANDARD 20
        USE_FOLDERS ON
        FOLDER Dependency)
# Set Include Folder
target_include_directories(${PROJECT_NAME} PUBLIC include)