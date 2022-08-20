module;
#include <cstdint>
#include <cmath>
#include <algorithm>
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
		explicit operator Point2<U>() const { return Point2<U>((U)this->x, (U)this->y); }

		auto operator+(Vector2<T> const& a) const -> Point2 { return Point2{ this->x + a.x,this->y + a.y }; }
		auto operator-(Vector2<T> const& a) const -> Point2 { return Point2{ this->x - a.x,this->y - a.y }; }
	};


	export template <class T>
	inline auto max(Point2<T> const& a, Point2<T> const& b) noexcept -> Point2<T> {
		return Point2<T>{std::max(a.x, b.x), std::max(a.y, b.y)};
	}

	export template <class T>
	inline auto min(Point2<T> const& a, Point2<T> const& b) noexcept -> Point2<T> {
		return Point2<T>{std::min(a.x, b.x), std::min(a.y, b.y)};
	}

	export using point2 = Point2<float>;
	export using ipoint2 = Point2<int32_t>;
	export using upoint2 = Point2<uint32_t>;
}