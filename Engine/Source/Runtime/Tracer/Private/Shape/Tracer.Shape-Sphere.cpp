module;
#include <algorithm>
#include <cmath>
module Tracer.Shape:Sphere;
import Tracer.Shape;
import :Shape;

import Math.Vector;
import Math.Transform;
import Math.Geometry;
import Math.Common;
import Math.Trigonometric;
import Math.RoundingError;
import Math.EquationSolving;

namespace SIByL::Tracer
{
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
		Math::ray3 const& r,
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
		Math::efloat b = Math::efloat(2.f) *(dx * ox + dy * oy + dz * oz);
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
		}
		// find parametric representation of sphere hit
		float u = phi / phiMax;
		float theta = std::acos(Math::clamp(pHit.z / radius, -1.f, 1.f));
		float v = (theta - thetaMin) / (thetaMax - thetaMin);
		//		compute sphere ∂p/∂u and ∂p/∂v
		//		compute sphere ∂n/∂u and ∂n/∂v
		// compute error bounds for sphere intersection

		// initialize SurfaceInteraction from parametric information

		// update tHit for quadric intersection
		*tHit = (float)tShapeHit;
		return true;
	}


}