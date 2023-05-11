#include "SE.Application.hpp"

namespace SIByL::Application {
Root::Root() {
  gMemManager.startUp();
  gLogManager.startUp();
  gEntityManager.startUp();
  gComponentManager.startUp();
  gResourceManager.startUp();
  // ext insertion
  gGfxManager.addExt<GFX::VideExtension>(GFX::Ext::VideoClip);
  gGfxManager.startUp();
}

Root::~Root() {
  gGfxManager.shutDown();
  gResourceManager.shutDown();
  gComponentManager.shutDown();
  gEntityManager.shutDown();
  gLogManager.shutDown();
  gMemManager.shutDown();
}

auto ApplicationBase::createMainWindow(
    Platform::WindowOptions const& options) noexcept -> void {
  mainWindow = Platform::Window::create(options);
}

auto ApplicationBase::run() noexcept -> void {
  // init the application
  Init();
  static double accumulatedTime = 0;
  // run the main loop
  while (!ShouldExit) {
    // if (Root::get()->startFrame()) {
    //	terminate();
    //	break;
    // }
    //  fetch main window events
    mainWindow->fetchEvents();
    double deltaTime = timer.deltaTime();
    accumulatedTime += deltaTime;
    while (accumulatedTime > FixedUpdateDelta) {
      FixedUpdate();
      accumulatedTime -= FixedUpdateDelta;
    }
    timer.update();
    Update(timer.deltaTime());

    mainWindow->endFrame();
    // Update window status, to check whether should exit
    ShouldExit |= !mainWindow->isRunning();
  }
  Exit();
  mainWindow->destroy();
}
}  // namespace SIByL::Application