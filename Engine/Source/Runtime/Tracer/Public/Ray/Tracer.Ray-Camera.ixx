export module Tracer.Ray:Camera;
import :Ray;
import :RayDifferential;
import :Sampler;
import :Medium;
import Tracer.Film;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
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