module;
export module Tracer.Camera:OrthographicCamera;
import :ProjectiveCamera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
	export struct OrthographicCamera :public ProjectiveCamera
	{
		OrthographicCamera(
			Math::AnimatedTransform const& cameraToWorld,
			Math::bounds2 const& screenWindow,
			float shutterOpen, float shutterClose,
			float lensr, float focald,
			Film* film, Medium const* medium);

		virtual auto generateRay(CameraSample const& sample, Ray* ray) const noexcept -> float override;

		auto generateRayDifferential(CameraSample const& sample, RayDifferential* rd) noexcept -> float;

		Math::vec3 dxCamera, dyCamera;
	};
}