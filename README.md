# SIByLEngine 2023.0

`SIByLEngine` is a personal toy game engine by [@SuikaSibyl](https://github.com/SuikaSibyl). I am planning to keep working on it for a long time and implement lots of realtime & offline effects in it.

Previous version could be find in: [SibylEngine2021](https://github.com/SuikaSibyl/SibylEngine2021). Version 2023 is a complete refactoring including better and more extendable structure, solution, design.

![EditorDemoImg](https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/SE2023_0_demo_ui.png)

## Builds

For now, no build tool has been set up. Just open `SIByLEngine.sln` with `VisualStudio 2022` on Windows 10/11.

By default, Nvidia GPU with Turing or higher architecture is required to correctly run the engine, as vulkan hardware raytracing is defaultly used. In the future, better compatibility with CPU raytracing fallback might be supported.

## Features
- Core
  - Fully using C++ 20 `Module` modern solution.
  - [Math module](./docs/SIByLDocument_002_Math.md) from scratch with SIMD acceleration.
  - [ECS module](./docs/SIByLDocument_003_ECS.md) from scratch (experimental).
  - [Resource module](./docs/SIByLDocument_004_Resource.md) from scratch (experimental).
- Rendering
  - [RHI module](./docs/SIByLDocument_004_RHI.md), with `WebGPU`-style API (only Vulkan backend supported now).
  - [Tracer module](./docs/SIByLDocument_005_Tracer.md) (CPU/GPU implementation are not integrated yet)
    - GPU based raytracing
      - Realtime Denoiser
        - Local Frequency Analysis Based Approaches [[summary](https://suikasibyl.github.io/CSE274-RealtimeDenoiser-WebPage/)]
          - [*AAF Softshadow*](http://graphics.berkeley.edu/papers/UdayMehta-AAF-2012-12/)
          - [*AAF Global Illumination*](https://cseweb.ucsd.edu/~ravir/filtering_GI_final.pdf)
          - [*MAAF Combined Effects*](https://cseweb.ucsd.edu/~ravir/paper_maaf.pdf)
    - CPU based raytracing (now only a core subset of PBRT)
  - [GFX module](./docs/SIByLDocument_006_GFX.md)
    - Basic Components & Resources
    - Render Dependency Graph System (experimental).
  - [SRenderer module](./docs/SIByLDocument_008_SRenderer.md)
    - Uniform scene packing for both rasterizer & ray tracer.
    - Procedure sphere primitive for ray tracing pipeline
    - Path Tracer (on going ...)
- Physics
  - // TODO.
- Editor Toolchain
  - `Dear ImGui` based editor.

## Dependencies
- `glad`: [Multi-Language Vulkan/GL/GLES/EGL/GLX/WGL Loader-Generator based on the official specs.](https://github.com/Dav1dde/glad)
- `glfw`: [A multi-platform library for OpenGL, OpenGL ES, Vulkan, window and input.](https://github.com/glfw/glfw)
- `Dear ImGui`: [Bloat-free Graphical User interface for C++ with minimal dependencies](https://github.com/ocornut/imgui)
- `stb`: [stb single-file public domain libraries for C/C++](https://github.com/nothings/stb)
- `tinyobjloader`: [Tiny but powerful single file wavefront obj loader](https://github.com/tinyobjloader/tinyobjloader)
- `tinygltf`: [Header only C++11 tiny glTF 2.0 library](https://github.com/syoyo/tinygltf)
- `vulkan`: [A low-overhead, cross-platform API, open standard for 3D graphics and computing](https://www.vulkan.org/)
- `vma`: [Easy to integrate Vulkan memory allocation library](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
