module;
#include <algorithm>
#include <cmath>
module Tracer.Shape:Sphere;
import Tracer.Shape;
import :Shape;

import Math.Transform;
import Math.Geometry;
import Math.Common;
import Math.Trigonometric;

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
		Math::ray3 const& ray,
		float* tHit,
		SurfaceInteraction* isect,
		bool testAlphaTexture) const -> bool
	{
		float phi;
		Math::point3 pHit;
		// Transform ray to object space
		//Math::vec3 oErr, dErr;
		//Math::ray3 = (*worldToObject) * (ray, &oErr, &dErr);
		// Compute quadric sphere coefficients

		// solve quadric equation for t values

		// compute sphere hit position and φ

		// test sphere intersection against clipping parameters

		// find parametric representation of sphere hit

		// compute error bounds for sphere intersection

		// initialize SurfaceInteraction from parametric information

		// update tHit for quadric intersection
		return true;
	}


}