module;
#include <limits>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
export module Tracer.Ray;
import SE.Core.Memory;
import SE.Math.Geometric;
import SE.Math.Misc;
import Tracer.Spectrum;
import Tracer.Film;
import Tracer.Base;

namespace SIByL::Tracer
{
	struct Ray;
	struct RayDifferential;
	struct Medium;

	export struct Ray
	{
		Ray() :tMax(Math::float_infinity), time(0.f), medium(nullptr) {}
		Ray(Math::point3 const& o, Math::vec3 const& d, float tMax = Math::float_infinity, float time = 0.f, Medium const* medium = nullptr)
			:o(o), d(d), tMax(tMax), time(time), medium(medium) {}

		/** origin */
		Math::point3 o;
		/** direction */
		Math::vec3 d;
		/** restrict the ray to segment [0,r(tMax)]*/
		mutable float tMax;
		/** time, used to handle animated scene*/
		float time;
		/** A medium containing its origin */
		Medium const* medium;

		operator Math::ray3() const { return Math::ray3{ o,d,tMax }; }
		Math::point3 operator()(float t) const { return o + d * t; }
	};

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

	export struct Medium
	{
		virtual auto Tr(Ray const& ray, Sampler& sampler) const noexcept -> Spectrum = 0;
	};

	export inline auto getMediumScatteringProperties(std::string const& name, Spectrum* sigma_a, Spectrum* sigma_s) noexcept -> bool {
		// TODO
		return false;
	}

	export struct Camera
	{
		Camera(Math::AnimatedTransform const& cameraToWorld, float shutterOpen,
			float shutterClose, Film* film, Medium const* medium);

		/**
		@brief	Compute the ray corresponding to a given sample
		@param	ray: Returned ray which is normalized
		@return How much the radiance arriving at the film plane along the ray
				will contribute to the final image
		*/
		virtual auto generateRay(CameraSample const& sample, Ray* ray) const noexcept -> float = 0;

		/**
		@brief	Compute a main raym but also corresponding rays for pixels
				shifted one pixel in x & y directions on the film plane
		@param	rd: Returned ray differential which dir member is normalized
		@return How much the radiance arriving at the film plane along the ray
				will contribute to the final image
		*/
		auto generateRayDifferential(CameraSample const& sample, RayDifferential* rd) const noexcept -> float;

		Math::AnimatedTransform cameraToWorld;
		float const shutterOpen, shutterClose;
		/** Handle image storage */
		Film* film;
		Medium const* medium;
	};
}

namespace SIByL::Math
{
	export inline auto operator*(Transform const& t, Tracer::Ray const& r)->Tracer::Ray;
	export inline auto operator*(AnimatedTransform const& t, Tracer::Ray const& r)->Tracer::Ray;
}
