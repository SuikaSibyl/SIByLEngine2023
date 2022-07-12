module;
#include <cmath>
#include <limits>
#include <cstdint>
export module Math.Geometry:Bounds2;
import Math.Vector;
import :Point2;

namespace SIByL::Math
{
	export template <class T>
		struct Bounds2
	{
		Bounds2();
		Bounds2(Point2<T> const& p);
		Bounds2(Point2<T> const& p1, Point2<T> const& p2);

		auto operator[](uint32_t i) const -> Point2<T> const&;
		auto operator[](uint32_t i) -> Point2<T>&;

		auto corner(uint32_t c) const -> Point2<T>;
		auto diagonal() const -> Vector2<T>;
		auto surfaceArea() const -> T;
		auto maximumExtent() const -> uint32_t;
		auto lerp(Point2<T> const& t) const -> Point2<T>;
		auto offset(Point2<T> const& p) const -> Vector2<T>;
		auto boundingCircle(Point2<T>* center, float* radius) const -> void;

		Point2<T> pMin;
		Point2<T> pMax;
	};

	// specialized alias
	// -----------------------
	export using bounds2 = Bounds2<float>;
	export using ibounds2 = Bounds2<int32_t>;

	// iterator for ibounds2
	// -----------------------
	class ibounds2Iterator : public std::forward_iterator_tag {
	public:
		ibounds2Iterator(const ibounds2& b, const ipoint2& pt)
			: p(pt), bounds(&b) {}

		auto operator++()->ibounds2Iterator;
		auto operator++(int)->ibounds2Iterator;
		auto operator==(const ibounds2Iterator& bi) const -> bool;
		auto operator!=(const ibounds2Iterator& bi) const -> bool;
		auto operator*() const noexcept -> ipoint2 { return p; }

	private:
		auto advance() noexcept -> void;
		ipoint2 p;
		ibounds2 const* bounds;
	};
	export inline auto begin(ibounds2 const& b)->ibounds2Iterator;
	export inline auto end(ibounds2 const& b)->ibounds2Iterator;

	// template impl
	// -----------------------
	template <class T>
	Bounds2<T>::Bounds2()
	{
		T minNum = std::numeric_limits<T>::lowest();
		T maxNum = std::numeric_limits<T>::max();

		pMin = Vector2<T>(maxNum);
		pMax = Vector2<T>(minNum);
	}

	template <class T>
	Bounds2<T>::Bounds2(Point2<T> const& p)
	{
		pMin = p;
		pMax = p;
	}
	
	template <class T>
	Bounds2<T>::Bounds2(Point2<T> const& p1, Point2<T> const& p2)
	{
		pMin = Vector2<T>(std::min(p1.x,p2.x), std::min(p1.y,p2.y));
		pMax = Vector2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
	}

	template <class T>
	auto Bounds2<T>::operator[](uint32_t i) const -> Point2<T> const&
	{
		return (i == 0) ? pMin : pMax;
	}

	template <class T>
	auto Bounds2<T>::operator[](uint32_t i) -> Point2<T>&
	{
		return (i == 0) ? pMin : pMax; 
	}

	template <class T>
	auto Bounds2<T>::corner(uint32_t c) const -> Point2<T>
	{
		return Point2<T>(
			(*this)[(c & 1)].x,
			(*this)[(c & 2) ? 1 : 0].y
			);
	}

	template <class T>
	auto Bounds2<T>::diagonal() const -> Vector2<T>
	{
		return pMax - pMin;
	}
	
	template <class T>
	auto Bounds2<T>::surfaceArea() const -> T
	{
		Vector2<T> d = diagonal();
		return d.x * d.y;
	}

	template <class T>
	auto Bounds2<T>::maximumExtent() const-> uint32_t
	{
		Vector3<T> d = diagonal();
		if (d.x > d.y)
			return 0;
		else
			return 1;
	}
	
	template <class T>
	auto Bounds2<T>::lerp(Point2<T> const& t) const -> Point2<T>
	{
		return Point2<T>({
			std::lerp(pMin.x, pMax.x, t.x),
			std::lerp(pMin.y, pMax.y, t.y)
			});
	}
	
	template <class T>
	auto Bounds2<T>::offset(Point2<T> const& p) const -> Vector2<T>
	{
		Vector2<T> o = p - pMin;
		if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
		if (pMax.y > pMin.y) o.x /= pMax.y - pMin.y;
		return o;
	}

	template <class T>
	auto Bounds2<T>::boundingCircle(Point2<T>* center, float* radius) const -> void
	{
		*center = (pMin + pMax) / 2;
		*radius = inside(*center, *this) ? distance(*center, pMax) : 0;
	}

	export template <class T>
	inline auto unionPoint(Bounds2<T> const& b, Point2<T> const& p) noexcept -> Bounds2<T>
	{
		return Bounds2<T>(Point2<T>(std::min(b.pMin.x, p.x),
									std::min(b.pMin.y, p.y)),
						  Point2<T>(std::max(b.pMax.x, p.x),
									std::max(b.pMax.y, p.y)));
	}

	export template <class T>
	inline auto unionBounds(Bounds2<T> const& b1, Bounds2<T> const& b2) noexcept -> Bounds2<T>
	{
		return Bounds2<T>(Point2<T>(std::min(b1.pMin.x, b2.pMin.x),
									std::min(b1.pMin.y, b2.pMin.y)),
					      Point2<T>(std::max(b1.pMax.x, b2.pMax.x),
									std::max(b1.pMax.y, b2.pMax.y)));
	}

	export template <class T>
	inline auto intersect(Bounds2<T> const& b1, Bounds2<T> const& b2) noexcept -> Bounds2<T>
	{
		return Bounds2<T>(Point2<T>(std::max(b1.pMin.x, b2.pMin.x),
									std::max(b1.pMin.y, b2.pMin.y)),
						  Point2<T>(std::min(b1.pMax.x, b2.pMax.x),
									std::min(b1.pMax.y, b2.pMax.y)));
	}

	export template <class T>
	inline auto overlaps(Bounds2<T> const& b1, Bounds2<T> const& b2) noexcept -> Bounds2<T>
	{
		bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
		bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
		return (x && y);
	}

	export template <class T>
	inline auto inside(Point2<T> const& p, Bounds2<T> const& b) noexcept -> bool
	{
		return (p.x >= b.pMin.x && p.x <= b.pMax.x&&
				p.y >= b.pMin.y && p.y <= b.pMax.y);
	}

	export template <class T>
	inline auto insideExclusive(Point2<T> const& p, Bounds2<T> const& b) noexcept -> bool
	{
		return (p.x >= b.pMin.x && p.x < b.pMax.x&&
				p.y >= b.pMin.y && p.y < b.pMax.y);
	}

	export template <class T>
	inline auto expand(Bounds2<T> const& b, float delta) noexcept -> bool
	{
		return Bounds2<T>(b.pMin - Vector2<T>(delta),
						  b.pMax + Vector2<T>(delta));
	}

	// impl
	// =======================
	auto ibounds2Iterator::operator++() -> ibounds2Iterator
	{
		advance();
		return *this;
	}

	auto ibounds2Iterator::operator++(int) -> ibounds2Iterator
	{
		ibounds2Iterator old = *this;
		advance();
		return old;
	}

	auto ibounds2Iterator::operator==(const ibounds2Iterator& bi) const -> bool
	{
		return p == bi.p && bounds == bi.bounds;
	}

	auto ibounds2Iterator::operator!=(const ibounds2Iterator& bi) const -> bool
	{
		return p != bi.p || bounds != bi.bounds;
	}

	auto ibounds2Iterator::advance() noexcept -> void
	{
		++p.x;
		if (p.x == bounds->pMax.x)
		{
			p.x = bounds->pMin.x;
			++p.y;
		}
	}

	inline auto begin(ibounds2 const& b) -> ibounds2Iterator
	{
		return ibounds2Iterator(b, b.pMin);
	}

	inline auto end(ibounds2 const& b) -> ibounds2Iterator
	{
		// Normally, the ending point is at the minimum x value and one past
		// the last valid y value.
		ipoint2 pEnd = ivec2(b.pMin.x, b.pMax.y);
		// However, if the bounds are degenerate, override the end point to
		// equal the start point so that any attempt to iterate over the bounds
		// exits out immediately.
		if (b.pMin.x >= b.pMax.x || b.pMin.y >= b.pMax.y)
			pEnd = b.pMin;
		return ibounds2Iterator(b, pEnd);
	}
}