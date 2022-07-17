module;
module Tracer.Camera:OrthographicCamera;
import Tracer.Camera;
import :ProjectiveCamera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import Math.Common;
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

	auto OrthographicCamera::generateRay(CameraSample const& sample, Ray* ray) const noexcept -> float
	{
		// Compute raster and camera sample positions
		Math::point3 pFilm{ sample.pFilm.x,sample.pFilm.y,0 };
		Math::point3 pCamera = rasterToCamera * pFilm;

		*ray = Ray(pCamera, Math::vec3{ 0,0,1 });
		// Modify ray for depth of field
		ray->time = Math::lerp(sample.time, shutterOpen, shutterClose);
		ray->medium = medium;
		*ray = cameraToWorld * (*ray);
		return 0.0f;
	}
}