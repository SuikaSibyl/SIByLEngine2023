module;
#include <algorithm>
export module Tracer.Shape:Sphere;
import :Shape;

import Math.Transform;
import Math.Geometry;
import Math.Common;

namespace SIByL::Tracer
{
	// Sphere is a special case of quadrics

	export struct Sphere :public Shape
	{
		Sphere(Math::Transform const* objectToWorld, Math::Transform const* worldToObject, bool reverseOrientation,
			float radius, float zMin, float zMax, float phiMax)
			: Shape(objectToWorld, worldToObject, reverseOrientation)
			, radius(radius)
			, zMin(Math::clamp(std::min(zMin, zMax), -radius, radius))
		{}

		float radius;
		float zMin;
		float zMax;
		float thetaMin;
		float thetaMax;
		float phiMax;
	};
}