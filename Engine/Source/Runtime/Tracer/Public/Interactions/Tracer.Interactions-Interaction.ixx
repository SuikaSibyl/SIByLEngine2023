export module Tracer.Interactions:Interaction;
import Math.Vector;
import Math.Geometry;
import Tracer.Medium;

namespace SIByL::Tracer
{
	export struct Interaction
	{
		//Interaction(Math::point3 const& p, Math::normal3 const& n, Math::vec3 const& pError,
		//	Math::vec3 const& wo, float time,
		//	const MediumInterface& mediumInterface)
		//	: p(p), time(time), pError(pError), wo(wo), n(n),
		//	mediumInterface(mediumInterface) { }

		auto isSurfaceInteraction() const noexcept -> bool;

		/** interaction position */
		Math::point3 p;
		/** interaction time */
		float time;
		/** a conservative bound on floating-point error of point p, when computed by ray intersection */
		Math::vec3 pError;
		/** if interaction lies along a ray, wo is the negative ray direction */
		Math::vec3 wo;
		/** surface normal of point p */
		Math::normal3 n;
		/** the scattering media at the point (if any) */

		// TODO::MediumInterface
	};
}