export module GFX.Geometric:Shape;
import Math.Transform;
import Math.Geometry;

namespace SIByL::GFX
{
	export struct Shape
	{
		virtual auto objectBound() const noexcept -> Math::bounds3 = 0;

		auto worldBound() const noexcept -> Math::bounds3;

		Math::Transform* objectToWorld;
		Math::Transform* worldToObject;
		bool const reverseOrientation;
		bool const transformSwapHandedness;
	};

	auto Shape::worldBound() const noexcept -> Math::bounds3 {
		return (*objectToWorld) * (objectBound());
	}

}