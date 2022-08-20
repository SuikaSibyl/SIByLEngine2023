module;
module Tracer.Camera:Camera;
import Tracer.Camera;
import Tracer.Medium;
import Tracer.Film;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
	Camera::Camera(Math::AnimatedTransform const& cameraToWorld, float shutterOpen,
		float shutterClose, Film* film, Medium const* medium)
		: cameraToWorld(cameraToWorld)
		, shutterOpen(shutterOpen)
		, shutterClose(shutterClose)
		, film(film)
		, medium(medium)
	{}

	auto Camera::generateRayDifferential(CameraSample const& sample, RayDifferential* rd) const noexcept -> float {
		float wt = generateRay(sample, rd);
		// Find camera ray after shifting one pixel in the x direction
		CameraSample sshift = sample;
		sshift.pFilm.x++;
		Ray rx;
		float wtx = generateRay(sshift, &rx);
		if (wtx == 0) return 0;
		rd->rxOrigin = rx.o;
		rd->rxDirection = rx.d;
		// Find camera ray after shifting one pixel in the y direction
		sshift = sample;
		sshift.pFilm.y++;
		Ray ry;
		float wty = generateRay(sshift, &ry);
		if (wty == 0) return 0;
		rd->ryOrigin = ry.o;
		rd->ryDirection = ry.d;

		rd->hasDifferentials = true;
		return wt;
	}
	
	auto Camera::generateRay(CameraSample const& sample, Ray* ray) const noexcept -> float {
		return 1.0f;
	}

}