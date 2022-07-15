module;
module Tracer.Camera:OrthographicCamera;
import :ProjectiveCamera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
	OrthographicCamera::OrthographicCamera(
		Math::AnimatedTransform const& cameraToWorld,
		Math::bounds2 const& screenWindow,
		float shutterOpen, float shutterClose,
		float lensr, float focald,
		Film* film, Medium const* medium)
		: ProjectiveCamera(cameraToWorld, Math::orthographic(0, 1),
			screenWindow, shutterOpen, shutterClose,
			lensr, focald, film, medium)
	{
		// Compute differential changes in origin for orthographic camera rays
		dxCamera = rasterToCamera * Math::point3{ 1,0,0 };
		dyCamera = rasterToCamera * Math::point3{ 0,1,0 };
	}
}