# SIByLEngine 2023.1

`SIByLEngine` is a personal toy game engine by [@SuikaSibyl](https://github.com/SuikaSibyl). I am planning to keep working on it for a long time and implement lots of realtime & offline effects in it.

Previous version could be find in: [SibylEngine2021](https://github.com/SuikaSibyl/SibylEngine2021). Version 2023 is a complete refactoring including better and more extendable structure, solution, design.

![EditorDemoImg](https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/SE2023_0_demo_ui.png)

## Builds

For now, no build tool has been set up. Just open `SIByLEngine.sln` with `VisualStudio 2022` on Windows 10/11.

By default, Nvidia GPU with Turing or higher architecture is required to correctly run the engine, as vulkan hardware raytracing is defaultly used. In the future, better compatibility with CPU raytracing fallback might be supported.

## Features
- Using C++ 20 `Module` for the whole engine.
- ...

## Modules
- ### Core Modules
  - [Math module](./docs/SIByLDocument_002_Math.md) from scratch with SIMD acceleration.
  - [ECS module](./docs/SIByLDocument_003_ECS.md) from scratch (experimental).
  - [Resource module](./docs/SIByLDocument_004_Resource.md) from scratch (experimental).
- ### Rendering Modules
  - [SE.RHI](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#rhi-module): Abstract layer with `WebGPU`-style API supporting `Vulkan` backend.
  - [SE.Tracer](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#tracer-module): `CPU ray tracing` implementation based on PBRT v3 (`GPU` implementation see `SE.SRenderer`)
  - [SE.Image](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#image-module): `Image` loader/storer for various format.
  - [SE.Video](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#video-module): `Video` loader/storer for various format, decoding via `ffmpeg`.
  - [SE.GFX](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#gfx-module): Higher-Level infrastructure on top pf `SE.RHI`
    - Defining `scene`, with many basic `component`s and  `resource`s.
    - Providing `serializing` & `deserializing` via `yaml-cpp`.
  - [SE.RDG](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Render-Dependency-Graph): `Render Dependency Graph` System (experimental).
    - `Pass`: rhi-pipeline-level (rasterize/fullscreen/compute/raytrace) atomic node in graph, with automatic pipeline and resource binding layouts creation via `spirv-cross` reflection of SPIR-V shaders.
    - `Graph`: a directed acyclic graph of `Pass`es, automatic resource management and barrier insertion.
    - `Pipeline`: a interface enabling cross-frame switching between multiple  `Graph`s.
  - [SE.SRenderer](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/SIByL-Renderer): Retained mode renderer based on SE.RDG
    - `Retained mode rendering` infrastructure for both rasterization and ray tracing
    - Providing lots of `RDG Pass/Graph/Pipeline` implementation.

- ### Physics
  - // TODO.
- ### Editor
  - `Editor_Core module`: ImGui-Vulkan interoperate and extensible GUI framework.
  - `Editor_GFX module`: Provides many editor widgets for SIByL Engine. 
  - `Editor_RDG module`: Provides DebugDraw system via RDG and RDG Viewer (no RDG editor yet).

![TracerDemo](https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/3tracer.png)

## Dependencies
- `glad`: [Multi-Language Vulkan/GL/GLES/EGL/GLX/WGL Loader-Generator based on the official specs.](https://github.com/Dav1dde/glad)
- `glfw`: [A multi-platform library for OpenGL, OpenGL ES, Vulkan, window and input.](https://github.com/glfw/glfw)
- `dear_imgui`: [Bloat-free Graphical User interface for C++ with minimal dependencies](https://github.com/ocornut/imgui)
- `ffmpeg`: [A complete, cross-platform solution to record, convert and stream audio and video.](https://ffmpeg.org/)
- `slang`: [Making it easier to work with shaders.](https://github.com/shader-slang/slang)
- `spirv-cross`: [A practical tool and library for performing reflection on SPIR-V and disassembling SPIR-V back to high level languages.](https://github.com/KhronosGroup/SPIRV-Cross)
- `stb`: [stb single-file public domain libraries for C/C++](https://github.com/nothings/stb)
- `tinygltf`: [Header only C++11 tiny glTF 2.0 library](https://github.com/syoyo/tinygltf)
- `tinyobjloader`: [Tiny but powerful single file wavefront obj loader](https://github.com/tinyobjloader/tinyobjloader)
- `vma`: [Easy to integrate Vulkan memory allocation library](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- `vulkan`: [A low-overhead, cross-platform API, open standard for 3D graphics and computing](https://www.vulkan.org/)
- `yaml-cpp`: [A YAML parser and emitter in C++](https://github.com/jbeder/yaml-cpp)