export module Tracer.Interactable:Interaction;
import Math.Vector;
import Math.Geometry;
import Tracer.Medium;
import Tracer.Ray;

namespace SIByL::Tracer
{
	export struct Interaction
	{
		Interaction() = default;
		Interaction(Math::point3 const& p, Math::normal3 const& n, Math::vec3 const& pError,
			Math::vec3 const& wo, float time,
			MediumInterface const& mediumInterface)
			: p(p), time(time), pError(pError), wo(wo), n(n),
			mediumInterface(mediumInterface) { }

		auto isSurfaceInteraction() const noexcept -> bool;

		auto spawnRay(Math::vec3 const& d) const noexcept -> Ray;
		auto spawnRayTo(Math::point3 const& p) const noexcept -> Ray;
		auto spawnRayTo(Interaction const& i) const noexcept -> Ray;

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
		MediumInterface mediumInterface;
	};
}