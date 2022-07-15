export module Tracer.Camera:Camera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
	export struct CameraSample {
		Math::point2 pFilm;
		Math::point2 pLens;
		float time;
	};

	export struct Camera
	{
		Camera(Math::AnimatedTransform const& cameraToWorld, float shutterOpen,
			float shutterClose, Film* film, Medium const* medium);

		/*!
		@brief	Compute the ray corresponding to a given sample
		@param	ray: Returned ray which is normalized
		@return How much the radiance arriving at the film plane along the ray
				will contribute to the final image
		*/
		virtual auto generateRay(CameraSample const& sample, Ray* ray) const noexcept -> float = 0;

		/*!
		@brief	Compute a main raym but also corresponding rays for pixels 
				shifted one pixel in x & y directions on the film plane
		@param	rd: Returned ray differential which dir member is normalized
		@return How much the radiance arriving at the film plane along the ray
				will contribute to the final image
		*/
		auto generateRayDifferential(CameraSample const& sample, RayDifferential* rd) noexcept -> float;

		Math::AnimatedTransform cameraToWorld;
		float const shutterOpen, shutterClose;
		Film* film;
		Medium const* medium;
	};
}