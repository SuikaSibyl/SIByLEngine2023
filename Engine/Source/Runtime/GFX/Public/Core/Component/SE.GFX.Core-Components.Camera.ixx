module;
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
export module SE.GFX.Core:Camera;
import SE.Core.ECS;
import SE.Core.Resource;
import SE.Math.Misc;
import SE.Math.Geometric;
import SE.RHI;
import :SerializeUtils;

namespace SIByL::GFX
{
	export struct CameraComponent {
		auto getViewMat() noexcept -> Math::mat4;
		auto getProjectionMat() noexcept -> Math::mat4;

		enum struct ProjectType {
			PERSPECTIVE,
			ORTHOGONAL,
		};

		/** serialize */
		static auto serialize(void* emitter, Core::EntityHandle const& handle) -> void;
		/** deserialize */
		static auto deserialize(void* compAoS, Core::EntityHandle const& handle) -> void;

		float fovy = 45.f;
		float aspect = 1;
		float near = 0.1f;
		float far = 100.0f;

		float left_right = 0;
		float bottom_top = 0;
		ProjectType projectType = ProjectType::PERSPECTIVE;

	private:
		Math::mat4 view;
		Math::mat4 projection;
	};

	auto CameraComponent::getViewMat() noexcept -> Math::mat4 {
		return view;
	}

	auto CameraComponent::getProjectionMat() noexcept -> Math::mat4 {
		if (projectType == ProjectType::PERSPECTIVE) {
			projection = Math::perspective(fovy, aspect, near, far).m;
		}
		else if (projectType == ProjectType::ORTHOGONAL) {
			projection = Math::orthographic(near, far).m;
		}
		return projection;
	}

	auto CameraComponent::serialize(void* pemitter, Core::EntityHandle const& handle) -> void {
		YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
		Core::Entity entity(handle);
		CameraComponent* camera = entity.getComponent<CameraComponent>();
		if (camera != nullptr) {
			emitter << YAML::Key << "CameraComponent";
			emitter << YAML::Value << YAML::BeginMap;
			emitter << YAML::Key << "fovy" << YAML::Value << camera->fovy;
			emitter << YAML::Key << "aspect" << YAML::Value << camera->aspect;
			emitter << YAML::Key << "near" << YAML::Value << camera->near;
			emitter << YAML::Key << "far" << YAML::Value << camera->far;
			emitter << YAML::Key << "ProjectType" << YAML::Value << (uint32_t)camera->projectType;
			emitter << YAML::EndMap;
		}
	}

	auto CameraComponent::deserialize(void* compAoS, Core::EntityHandle const& handle) -> void {
		YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
		Core::Entity entity(handle);
		auto cameraComponentAoS = components["CameraComponent"];
		if (cameraComponentAoS) {
			CameraComponent* camRef = entity.addComponent<CameraComponent>();
			camRef->fovy = cameraComponentAoS["fovy"].as<float>();
			camRef->aspect = cameraComponentAoS["aspect"].as<float>();
			camRef->near = cameraComponentAoS["near"].as<float>();
			camRef->far = cameraComponentAoS["far"].as<float>();
			camRef->projectType = (ProjectType)cameraComponentAoS["ProjectType"].as<uint32_t>();
		}
	}
}