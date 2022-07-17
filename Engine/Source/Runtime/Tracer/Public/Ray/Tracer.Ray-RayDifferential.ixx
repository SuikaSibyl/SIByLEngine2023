module;
#include <limits>
export module Tracer.Ray:RayDifferential;
import Math.Vector;
import Math.Geometry;
import Math.Limits;
import Tracer.Medium;
import :Ray;

namespace SIByL::Tracer
{
	/**
	* Could be used to estimate the projected area on the image plane of a small part of the scene
	* from which Texture could compute the average value over the area
	*/
	export struct RayDifferential :public Ray
	{
		RayDifferential() :hasDifferentials(false) {}
		RayDifferential(Math::point3 const& o, Math::vec3 const& d, float tMax = Math::float_infinity,
			float time = 0.f, Medium const* medium = nullptr)
			: Ray(o, d, tMax, time, medium)
			, hasDifferentials(false) {}
		RayDifferential(Ray const& ray)
			: Ray(ray)
			, hasDifferentials(false) {}

		/**
		* Camera defaultly compute rays which are spaced one pixel apart
		* Integrators could have multiple samples in a pixel, so use scaleDifferentials to
		* update differential rays for an estimated sample spacing of s
		*/
		auto scaleDifferentials(float s) noexcept -> void;

		// contains two auxiliary rays
		// they are offset by one sample in the x & y direction from the main ray on the film plane
		Math::point3 rxOrigin, ryOrigin;
		Math::vec3 rxDirection, ryDirection;
		bool hasDifferentials = false;
	};

	auto RayDifferential::scaleDifferentials(float s) noexcept -> void
	{
		rxOrigin = o + (rxOrigin - o) * s;
		ryOrigin = o + (ryOrigin - o) * s;
		rxDirection = d + (rxDirection - d) * s;
		ryDirection = d + (ryDirection - d) * s;
	}
}