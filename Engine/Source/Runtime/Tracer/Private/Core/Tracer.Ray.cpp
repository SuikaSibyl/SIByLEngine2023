module;
#include <limits>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
module Tracer.Ray;
import Core.Memory;
import Math.Vector;
import Math.Geometry;
import Math.Limits;
import Math.Transform;
import Math.Random;
import Tracer.Spectrum;
import Tracer.Medium;

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

namespace SIByL::Math
{
	inline auto operator*(Transform const& t, Tracer::Ray const& r)->Tracer::Ray {
		Math::ray3 ray = (Math::ray3)r;
		ray = t * ray;
		Tracer::Ray ret;
		ret.o = ray.o;
		ret.d = ray.d;
		ret.tMax = ray.tMax;
		ret.medium = r.medium;
		ret.time = r.time;
		return r;
	}

	inline auto operator*(AnimatedTransform const& t, Tracer::Ray const& r)->Tracer::Ray {
		Math::ray3 ray = (Math::ray3)r;
		Transform interpTrans;
		t.interpolate(r.time, &interpTrans);
		return interpTrans * r;
	}
}