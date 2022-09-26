set input=%1
set filename=%input:~0,-5%
glslc ../Engine/Shaders/%1 -O -o ../Engine/Binaries/Runtime/spirv/%filename%.spv --target-env=vulkan1.1