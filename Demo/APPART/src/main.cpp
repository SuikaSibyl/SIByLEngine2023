#include <crtdbg.h>
#include <glad/glad.h>
#include <imgui.h>
#include <tinygltf/tiny_gltf.h>

#include <Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarGTPass.hpp>
#include <Pipeline/SE.SRendere-ForwardPipeline.hpp>
#include <Pipeline/SE.SRendere-RTGIPipeline.hpp>
#include <Pipeline/SE.SRendere-SSRXPipeline.hpp>
#include <Plugins/SE.SRendererExt.GeomTab.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Application.hpp>
#include <SE.Editor.Config.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.Editor.DebugDraw.hpp>
#include <SE.SRenderer.hpp>
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <stack>
#include <typeinfo>

//#include "FLIPDemo.hpp"

int main() {
  // application root, control all managers
  Application::Root root;
  // run app
  //FLIPApplication app;
  //app.createMainWindow({Platform::WindowVendor::GLFW, L"SIByL Elastic Demo",
  //                      1920, 1080, Platform::WindowProperties::VULKAN_CONTEX});
  //app.run();
}