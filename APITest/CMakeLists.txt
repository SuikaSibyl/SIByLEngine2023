cmake_minimum_required(VERSION 3.23)
# Set project setting
set(PROJECT_NAME APITest)
set(CMAKE_CXX_STANDARD 20)
# Add target executable
file(GLOB_RECURSE INC_FILES "include/*.h")
file(GLOB_RECURSE SRC_FILES "src/*.h" "src/*.cpp")
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
        PRIVATE
        ${INC_FILES}
        ${SRC_FILES})
# Set c++ standard 20
# Set In Folder - Dependency
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CXX_STANDARD 20)
# Set Include Folder
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        yaml
        imgui
        HSL
        Math)