module;
module Tracer.Camera:PerspectiveCamera;
import :ProjectiveCamera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import Math.Common;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
	PerspectiveCamera::PerspectiveCamera(
		Math::AnimatedTransform const& cameraToWorld,
		Math::bounds2 const& screenWindow,
		float shutterOpen, float shutterClose,
		float lensr, float focald, float fov,
		Film* film, Medium const* medium)
		: ProjectiveCamera(cameraToWorld, Math::perspective(fov, 1e-2f, 1000.f),
			screenWindow, shutterOpen, shutterClose,
			lensr, focald, film, medium)
	{
		// compute differential changes in origin for perspective camera rays
		dxCamera = rasterToCamera * Math::point3{ 1,0,0 } - rasterToCamera * Math::point3{ 0,0,0 };
		dyCamera = rasterToCamera * Math::point3{ 0,1,0 } - rasterToCamera * Math::point3{ 0,0,0 };

		// compute image plane bounds at z=1 for perspective camera

	}
	
	auto PerspectiveCamera::generateRay(CameraSample const& sample, Ray* ray) const noexcept -> float {
		// Compute raster and camera sample positions
		Math::point3 pFilm{ sample.pFilm.x,sample.pFilm.y,0 };
		Math::point3 pCamera = rasterToCamera * pFilm;

		*ray = Ray(Math::point3{ 0,0,0 }, Math::normalize(Math::vec3(pCamera)));
		// Modify ray for depth of field
		ray->time = Math::lerp(sample.time, shutterOpen, shutterClose);
		ray->medium = medium;
		*ray = cameraToWorld * (*ray);
		return 1;
	}

	auto PerspectiveCamera::generateRayDifferential(CameraSample const& sample, RayDifferential* rd) noexcept -> float {
		
		
		return 1.0f;
	}
}