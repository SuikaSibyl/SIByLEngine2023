module;
#include <cmath>
export module SE.Editor.GFX:CameraController;
import SE.Platform.Window;
import SE.Core.Misc;
import SE.Math.Geometric;
import SE.Math.Spline;
import SE.GFX.Core;

namespace SIByL::Editor
{
	export struct CameraState {
		float yaw = 0;
		float pitch = 0;
		float roll = 0;

		float x = 0;
		float y = 0;
		float z = 0;

		auto setFromTransform(GFX::TransformComponent const& transform) noexcept -> void {
			Math::vec3 eulerAngles = transform.eulerAngles;
			Math::vec3 translation = transform.translation;

			pitch = eulerAngles.x;
			yaw = eulerAngles.y;
			roll = eulerAngles.z;

			x = translation.x;
			y = translation.y;
			z = translation.z;
		}

		auto lerpTowards(CameraState const& target, float positionLerpPct, float rotationLerpPct) noexcept -> void {
			yaw = std::lerp(yaw, target.yaw, rotationLerpPct);
			pitch = std::lerp(pitch, target.pitch, rotationLerpPct);
			roll = std::lerp(roll, target.roll, rotationLerpPct);

			x = std::lerp(x, target.x, positionLerpPct);
			y = std::lerp(y, target.y, positionLerpPct);
			z = std::lerp(z, target.z, positionLerpPct);
		}

		auto updateTransform(GFX::TransformComponent& transform) noexcept -> void {
			transform.eulerAngles = Math::vec3(pitch, yaw, roll);
			transform.translation = Math::vec3(x, y, z);
		}
	};

	export struct SimpleCameraController {
		SimpleCameraController() :input(nullptr), timer(nullptr) {}
		SimpleCameraController(Platform::Input* input, Core::Timer* timer) :input(input), timer(timer) {}

		auto inline init(Platform::Input* input, Core::Timer* timer) {
			this->input = input;
			this->timer = timer;
		}

		float mouseSensitivityMultiplier = 0.01f;
		CameraState targetCameraState;
		CameraState interpolatingCameraState;
		float boost = 3.5f;
		float positionLerpTime = 0.2f;
		float mouseSensitivity = 60.0f;
		Math::AnimationCurve mouseSensitivityCurve = { {0,0.5,0,5}, {1,2.5,0,0} };
		float rotationLerpTime = 0.01f;
		bool invertY = true;

		auto onEnable(GFX::TransformComponent const& transform) noexcept -> void {
			targetCameraState.setFromTransform(transform);
			interpolatingCameraState.setFromTransform(transform);
		}

		auto getInputTranslationDirection() noexcept -> Math::vec3 {
			Math::vec3 direction(0.0f, 0.0f, 0.0f);
			if (input->isKeyPressed(Platform::SIByL_KEY_W)) {
				direction += Math::vec3(0, 0, +1); // forward
			}
			if (input->isKeyPressed(Platform::SIByL_KEY_S)) {
				direction += Math::vec3(0, 0, -1); // back
			}
			if (input->isKeyPressed(Platform::SIByL_KEY_A)) {
				direction += Math::vec3(-1, 0, 0); // left
			}
			if (input->isKeyPressed(Platform::SIByL_KEY_D)) {
				direction += Math::vec3(1, 0, 0); // right
			}
			if (input->isKeyPressed(Platform::SIByL_KEY_Q)) {
				direction += Math::vec3(0, -1, 0); // down
			}
			if (input->isKeyPressed(Platform::SIByL_KEY_E)) {
				direction += Math::vec3(0, 1, 0); // up
			}
			return direction;
		}

		auto bindTransform(GFX::TransformComponent* transform) noexcept -> void {
			targetCameraState.setFromTransform(*transform);
			interpolatingCameraState.setFromTransform(*transform);
			this->transform = transform;
		}

		auto onUpdate() noexcept -> void {
			// rotation
			static bool justPressedMouse = true;
			static float last_x = 0;
			static float last_y = 0;
			if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_2)) {
				input->disableCursor();
				float x = input->getMouseX();
				float y = input->getMouseY();
				if (justPressedMouse) {
					last_x = x;
					last_y = y;
					justPressedMouse = false;
				}
				else {
					Math::vec2 mouseMovement = Math::vec2(x - last_x, y - last_y) * 0.0005f * mouseSensitivityMultiplier * mouseSensitivity;
					if (invertY)
						mouseMovement.y = -mouseMovement.y;
					last_x = x;
					last_y = y;

					float mouseSensitivityFactor = mouseSensitivityCurve.evaluate(mouseMovement.length());

					targetCameraState.yaw += mouseMovement.x * mouseSensitivityFactor;
					targetCameraState.pitch += mouseMovement.y * mouseSensitivityFactor;
				}
			}
			else if (!justPressedMouse) {
				input->enableCursor();
				justPressedMouse = true;
			}

			// translation
			Math::vec3 translation = getInputTranslationDirection();
			translation *= timer->deltaTime() * 0.1;

			// speed up movement when shift key held
			if (input->isKeyPressed(Platform::SIByL_KEY_LEFT_SHIFT)) {
				translation *= 10.0f;
			}

			// modify movement by a boost factor ( defined in Inspector and modified in play mode through the mouse scroll wheel)
			float y = input->getMouseScrollY();
			boost += y * 0.01f;
			translation *= powf(2.0f, boost);

			Math::vec3 rotatedFoward;
			rotatedFoward.x = std::cos((targetCameraState.yaw)) * std::cos((targetCameraState.pitch));
			rotatedFoward.y = std::sin((targetCameraState.pitch));
			rotatedFoward.z = std::sin((targetCameraState.yaw)) * std::cos((targetCameraState.pitch));
			Math::vec3 up = Math::vec3(0.0f, 1.0f, 0.0f);
			Math::vec3 cameraRight = Math::normalize(Math::cross(rotatedFoward, up));
			Math::vec3 cameraUp = Math::cross(cameraRight, rotatedFoward);
			Math::vec3 movement = translation.z * rotatedFoward + translation.x * cameraRight + translation.y * cameraUp;
			
			targetCameraState.x += movement.x;
			targetCameraState.y += movement.y;
			targetCameraState.z += movement.z;

			//targetCameraState.translate(translation);

			// Framerate-independent interpolation
			// calculate the lerp amount, such that we get 99% of the way to our target in the specified time
			float positionLerpPct = 1.f - expf(log(1.f - 0.99f) / positionLerpTime * timer->deltaTime());
			float rotationLerpPct = 1.f - expf(log(1.f - 0.99f) / rotationLerpTime * timer->deltaTime());
			interpolatingCameraState.lerpTowards(targetCameraState, positionLerpPct, rotationLerpPct);

			if (transform != nullptr) interpolatingCameraState.updateTransform(*transform);
		}

	private:
		GFX::TransformComponent* transform = nullptr;
		Platform::Input* input;
		Core::Timer* timer;
	};
}