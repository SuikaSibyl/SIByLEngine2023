module;
#include <algorithm>
export module Tracer.Shape:Sphere;
import :Shape;

import Math.Transform;
import Math.Geometry;
import Math.Common;

namespace SIByL::Tracer
{
	/**
	* Sphere is a special case of quadrics.
	* The quadrics here is: x^2 + y^2 + z^2 = r^2.
	*/
	export struct Sphere :public Shape
	{
		/**
		* Construct a sphere with possibly surface cut off by parameters.
		* @param zMin: min z value, so that sphere below zMin plane will be removed;
		* @param zMax: min z value, so that sphere below zMin plane will be removed;
		* @param phiMax: max phi value in degree, so that spherical phi value above this will be removed;
		*/
		Sphere(Math::Transform const* objectToWorld, Math::Transform const* worldToObject, bool reverseOrientation,
			float radius, float zMin, float zMax, float phiMax);

		/** @see Shape::objectBound() */
		virtual auto objectBound() const noexcept -> Math::bounds3 override;

		/** @see Shape::intersect() */
		virtual auto intersect(
			Math::ray3 const& ray,
			float* tHit,
			SurfaceInteraction* isect,
			bool testAlphaTexture = true) const -> bool override;

		/** @see Shape::area() */
		virtual auto area() const noexcept -> float override { return 0.f; }

		float const radius;
		float const zMin, zMax;

		/**
		* Parametric reprensentation of sphere
		* x = r sinθ cosφ 
		* y = r sinθ sinφ
		* z = r cosθ
		* Where θ∈[0,π]， φ∈[0,2π]
		*/
		float const thetaMin, thetaMax, phiMax;

		/**
		* φ = u φmax
		* θ = θmin + v(θmax − θmin).
		*/

	};
}