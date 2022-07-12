export module GFX.Geometric:Sphere;
import :Shape;

namespace SIByL::GFX
{
	export struct Sphere :public Shape
	{
	private:
		float radius;
	};
}