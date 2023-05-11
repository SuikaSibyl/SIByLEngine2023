#pragma once
#include <memory>
#include <typeinfo>
#include <Print/SE.Core.Log.hpp>
#include <Memory/SE.Core.Memory.hpp>
#include <ECS/SE.Core.ECS.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.GFX.hpp>
#include <SE.Video.hpp>

namespace SIByL::Application {
SE_EXPORT struct Root {
  Root();
  ~Root();

  Core::MemoryManager gMemManager;
  Core::LogManager gLogManager;
  Core::EntityManager gEntityManager;
  Core::ComponentManager gComponentManager;
  Core::ResourceManager gResourceManager;
  GFX::GFXManager gGfxManager;
};

SE_EXPORT struct ApplicationBase {
 public:
  // --------------------------
  // Entry Methods
  // --------------------------
  /** define the main window */
  auto createMainWindow(Platform::WindowOptions const& options) noexcept
      -> void;
  /** run the application */
  auto run() noexcept -> void;
  /** terminate the application */
  auto terminate() noexcept -> void { ShouldExit = true; }

  // --------------------------
  // Override Life Functions
  // --------------------------
  /** Initialize the application */
  virtual auto Init() noexcept -> void{};
  /** Update the application every loop */
  virtual auto Update(double deltaTime) noexcept -> void{};
  /** Update the application every fixed update timestep */
  virtual auto FixedUpdate() noexcept -> void{};
  /** Exit after the main loop ends */
  virtual auto Exit() noexcept -> void{};

  // --------------------------
  // Setting Parameters
  // --------------------------
  /** Fixed update timestep */
  static constexpr double const FixedUpdateDelta = 0.2;
  /** Whether the application should exit */
  bool ShouldExit = false;

  // --------------------------
  // Data
  // --------------------------
  std::unique_ptr<Platform::Window> mainWindow = nullptr;
  Core::Timer timer;
};
}  // namespace SIByL::Application