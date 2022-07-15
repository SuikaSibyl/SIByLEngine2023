module;
#include <cstdint>
export module Math.Geometry:Point2;
import Math.Vector;

namespace SIByL::Math
{
	export template <class T>
		struct Point2 :public Vector2<T>
	{
		Point2(Vector2<T> const& v = { 0,0 })
			:Vector2<T>(v) {}

		Point2(T const& x, T const& y)
			:Vector2<T>(x, y) {}

		template <typename U>
		explicit Point2(Point2<U> const& p)
			:Vector2<T>((T)p.x, (T)p.y) {}

		template <typename U>
		explicit operator Vector2<U>() const { return Vector2<U>(this->x, this->y); }
	};

	export using point2 = Point2<float>;
	export using ipoint2 = Point2<int32_t>;
	export using upoint2 = Point2<uint32_t>;
}