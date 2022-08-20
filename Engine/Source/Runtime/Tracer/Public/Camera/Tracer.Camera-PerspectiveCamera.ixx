module;
export module Tracer.Camera:PerspectiveCamera;
import :ProjectiveCamera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
	export struct PerspectiveCamera :public ProjectiveCamera
	{
		PerspectiveCamera(
			Math::AnimatedTransform const& cameraToWorld,
			Math::bounds2 const& screenWindow,
			float shutterOpen, float shutterClose,
			float lensr, float focald, float fov,
			Film* film, Medium const* medium);

		virtual auto generateRay(CameraSample const& sample, Ray* ray) const noexcept -> float override;

		auto generateRayDifferential(CameraSample const& sample, RayDifferential* rd) const noexcept -> float;

		Math::vec3 dxCamera, dyCamera;
	};
}