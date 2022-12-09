module;
#include <algorithm>
#include <cmath>
export module Tracer.Shape:Sphere;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Interactable;
import Tracer.Sampling;

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
			Ray const& ray,
			float* tHit,
			SurfaceInteraction* isect,
			bool testAlphaTexture = true) const -> bool override;

		/** @see Shape::area() */
		virtual auto area() const noexcept -> float override { return 0.f; }

		virtual auto sample(Math::point2 const& u) const noexcept -> Interaction override;


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


	Sphere::Sphere(Math::Transform const* objectToWorld, Math::Transform const* worldToObject, bool reverseOrientation,
		float radius, float zMin, float zMax, float phiMax)
		: Shape(objectToWorld, worldToObject, reverseOrientation)
		, radius(radius)
		, zMin(Math::clamp(std::min(zMin, zMax), -radius, radius))
		, zMax(Math::clamp(std::max(zMin, zMax), -radius, radius))
		, thetaMin(std::acos(Math::clamp(zMin / radius, -1.f, 1.f)))
		, thetaMax(std::acos(Math::clamp(zMax / radius, -1.f, 1.f)))
		, phiMax(Math::radians(Math::clamp(phiMax, 0.f, 360.f)))
	{}

	auto Sphere::objectBound() const noexcept -> Math::bounds3 {
		// get max sinθ value for xy boundary
		float sinThetaMax = 0.f;
		if (zMin <= 0.0f && zMax >= 0.0f) sinThetaMax = 1.0f;
		else if (zMax < 0.0f) sinThetaMax = std::sqrt(radius * radius - zMax * zMax);
		else sinThetaMax = std::sqrt(radius * radius - zMin * zMin);
		// compute xy boundary
		float radiusXsinTheta = radius * sinThetaMax;
		float xMin = (phiMax > Math::float_Pi) ? (-radiusXsinTheta) : (std::cos(phiMax) * radiusXsinTheta);
		float xMax = radiusXsinTheta;
		float yMin = 0.0f, yMax = radiusXsinTheta;
		if (phiMax <= 0.5 * Math::float_Pi)
			yMax = std::sin(phiMax) * radiusXsinTheta;
		else if (phiMax <= Math::float_Pi) {}
		else if (phiMax <= Math::float_Pi * 1.5)
			yMin = std::sin(phiMax) * radiusXsinTheta;
		else
			yMin = -radiusXsinTheta;
		// get the final bounds
		return Math::bounds3(
			Math::point3(xMin, yMin, zMin),
			Math::point3(xMax, yMax, zMax)
		);
	}

	auto Sphere::intersect(
		Ray const& r,
		float* tHit,
		SurfaceInteraction* isect,
		bool testAlphaTexture) const -> bool
	{
		float phi;
		Math::point3 pHit;
		// Transform ray to object space
		Math::vec3 oErr, dErr;
		Math::ray3 ray = (*worldToObject)(r, oErr, dErr);
		// Initialize efloat ray coordinate values
		Math::efloat ox(ray.o.x, oErr.x), oy(ray.o.y, oErr.y), oz(ray.o.z, oErr.z);
		Math::efloat dx(ray.d.x, dErr.x), dy(ray.d.y, dErr.y), dz(ray.d.z, dErr.z);
		// Compute quadric sphere coefficients
		Math::efloat a = dx * dx + dy * dy + dz * dz;
		Math::efloat b = Math::efloat(2.f) * (dx * ox + dy * oy + dz * oz);
		Math::efloat c = ox * ox + oy * oy + oz * oz - Math::efloat(radius) * Math::efloat(radius);
		// solve quadric equation for t values
		Math::efloat t0, t1;
		if (!quadratic(a, b, c, t0, t1))
			return false;
		// check quadric shape t0 & t1 for nearest intersection
		if (t0.upperBound() > ray.tMax || t1.lowerBound() <= 0)
			return false;
		Math::efloat tShapeHit = t0;
		if (tShapeHit.lowerBound() <= 0) {
			tShapeHit = t1;
			if (tShapeHit.upperBound() > ray.tMax)
				return false;
		}
		// compute sphere hit position and φ
		pHit = ray((float)tShapeHit);
		//		refine sphere intersection point
		if (pHit.x == 0 && pHit.y == 0) pHit.x = 1e-5f * radius;
		phi = std::atan2(pHit.y, pHit.x);
		if (phi < 0) phi += 2 * Math::float_Pi;
		// test sphere intersection against clipping parameters
		if ((zMin > -radius && pHit.z < zMin) ||
			(zMax < +radius && pHit.z > zMax) || phi > phiMax) {
			if (tShapeHit == t1) return false;
			if (t1.upperBound() > ray.tMax) return false;
			tShapeHit = t1;
			// compute sphere hit position and Φ
			pHit = ray((float)tShapeHit);
			if (pHit.x == 0 && pHit.y == 0) pHit.x = 1e-5f * radius;
			phi = std::atan2(pHit.y, pHit.x);
			if (phi < 0) phi += 2 * Math::float_Pi;
			if ((zMin > -radius && pHit.z < zMin) ||
				(zMax < +radius && pHit.z > zMax) || phi > phiMax)
				return false;
		}
		// find parametric representation of sphere hit
		float u = phi / phiMax;
		float theta = std::acos(Math::clamp(pHit.z / radius, -1.f, 1.f));
		float v = (theta - thetaMin) / (thetaMax - thetaMin);
		//		compute sphere ∂p/∂u and ∂p/∂v
		float zRadius = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
		float invZRadius = 1 / zRadius;
		float cosPhi = pHit.x * invZRadius;
		float sinPhi = pHit.y * invZRadius;
		Math::vec3 dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
		Math::vec3 dpdv = (thetaMax - thetaMin) * Math::vec3(pHit.z * cosPhi, pHit.z * sinPhi, -radius * std::sin(theta));
		//		compute sphere ∂n/∂u and ∂n/∂v
		Math::vec3 d2Pduu = -phiMax * phiMax * Math::vec3(pHit.x, pHit.y, 0);
		Math::vec3 d2Pduv = (thetaMax - thetaMin) * pHit.z * phiMax * Math::vec3(-sinPhi, cosPhi, 0.);
		Math::vec3 d2Pdvv = -(thetaMax - thetaMin) * (thetaMax - thetaMin) * Math::vec3(pHit.x, pHit.y, pHit.z);
		//  Compute coefficients for fundamental forms
		float E = Math::dot(dpdu, dpdu);
		float F = Math::dot(dpdu, dpdv);
		float G = Math::dot(dpdv, dpdv);
		Math::vec3 N = Math::normalize(Math::cross(dpdu, dpdv));
		float e = Math::dot(N, d2Pduu);
		float f = Math::dot(N, d2Pduv);
		float g = Math::dot(N, d2Pdvv);
		//  Compute ∂n / ∂u and ∂n / ∂v from fundamental form coefficients
		float invEGF2 = 1 / (E * G - F * F);
		Math::normal3 dndu = Math::normal3((f * F - e * G) * invEGF2 * dpdu + (e * F - f * E) * invEGF2 * dpdv);
		Math::normal3 dndv = Math::normal3((g * F - f * G) * invEGF2 * dpdu + (f * F - g * E) * invEGF2 * dpdv);
		// compute error bounds for sphere intersection
		Math::vec3 pError;
		// initialize SurfaceInteraction from parametric information
		*isect = (*objectToWorld) * (SurfaceInteraction(pHit, pError, Math::point2(u, v), -ray.d, dpdu, dpdv,
				dndu, dndv, r.time, this));
		// update tHit for quadric intersection
		*tHit = (float)tShapeHit;
		return true;
	}

	auto Sphere::sample(Math::point2 const& u) const noexcept -> Interaction {
		Math::point3 pObj = Math::point3(0, 0, 0) + radius * uniformSampleSphere(u);
		Interaction it;
		it.n = Math::normalize((*objectToWorld)*(Math::normal3(pObj.x, pObj.y, pObj.z)));
		if (reverseOrientation) it.n *= -1;
		// Reproject pObj to sphere surfaceand compute pObjError
		pObj *= radius / Math::distance(pObj, Math::point3(0, 0, 0));
		Math::vec3 pObjError = Math::gamma(5) * abs((Math::vec3)pObj);
		it.p = (*objectToWorld)(pObj, pObjError, it.pError);
		return it;
	}
}