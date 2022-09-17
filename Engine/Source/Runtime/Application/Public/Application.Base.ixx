export module Application.Base;
import Core.Memory;
import Application.Root;
import Platform.Window;

namespace SIByL::Application
{
	export struct ApplicationBase
	{
	public:
		// --------------------------
		// Entry Methods
		// --------------------------
		/** define the main window */
		auto createMainWindow(Platform::WindowOptions const& options) noexcept -> void;
		/** run the application */
		auto run() noexcept -> void;
		/** terminate the application */
		auto terminate() noexcept -> void { ShouldExit = true; }

		// --------------------------
		// Override Life Functions
		// --------------------------
		/** Initialize the application */
		virtual auto Init() noexcept -> void {};
		/** Update the application every loop */
		virtual auto Update() noexcept -> void {};
		/** Update the application every fixed update timestep */
		virtual auto FixedUpdate() noexcept -> void {};

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
		Scope<Platform::Window> mainWindow = nullptr;
	};

	auto ApplicationBase::createMainWindow(Platform::WindowOptions const& options) noexcept -> void {
		mainWindow = Platform::Window::create(options);
	}

	auto ApplicationBase::run() noexcept -> void {
		// init the application
		Init();
		// run the main loop
		while (!ShouldExit) {
			//if (Root::get()->startFrame()) {
			//	terminate();
			//	break;
			//}
			// fetch main window events
			mainWindow->fetchEvents();
			//double deltaTime = Root::get()->timer->deltaTime();
			//accumulatedTime += deltaTime;
			//while (accumulatedTime > fixedUpdateDelta) {
			//	FixedUpdate();
			//	accumulatedTime -= fixedUpdateDelta;
			//}
			//Update(deltaTime);
			Update();
			//Root::get()->endFrame();

			mainWindow->endFrame();
			// Update window status, to check whether should exit
			ShouldExit |= !mainWindow->isRunning();
		}
		mainWindow->destroy();
	}
}