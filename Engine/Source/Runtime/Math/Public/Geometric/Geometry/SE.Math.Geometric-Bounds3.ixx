module;
#include <cmath>
#include <limits>
#include <cstdint>
export module SE.Math.Geometric:Bounds3;
import :Vector3;
import :Point3;
import :Ray3;

namespace SIByL::Math
{
	export template <class T>
		struct Bounds3
	{
		Bounds3();
		Bounds3(Point3<T> const& p);
		Bounds3(Point3<T> const& p1, Point3<T> const& p2);

		auto operator[](uint32_t i) const -> Point3<T> const&;
		auto operator[](uint32_t i) -> Point3<T>&;

		auto corner(uint32_t c) const -> Point3<T>;
		auto diagonal() const -> Vector3<T>;
		auto surfaceArea() const -> T;
		auto volume() const -> T;
		auto maximumExtent() const -> uint32_t;
		auto lerp(Point3<T> const& t) const->Point3<T>;
		auto offset(Point3<T> const& p) const->Vector3<T>;
		auto boundingSphere(Point3<T>* center, float* radius) const -> void;
		
		/** Test a ray-AABB intersection */
		inline auto intersectP(ray3 const& ray, float* hitt0, float* hitt1) const -> bool;
		/** Test a ray-AABB intersection, using precomputed values indicating negativity of each components. */
		inline auto intersectP(ray3 const& ray, vec3 const& invDir, ivec3 dirIsNeg) const -> bool;

		Point3<T> pMin;
		Point3<T> pMax;
	};

	template <class T>
	Bounds3<T>::Bounds3()
	{
		T minNum = std::numeric_limits<T>::lowest();
		T maxNum = std::numeric_limits<T>::max();

		pMin = Vector3<T>(maxNum);
		pMax = Vector3<T>(minNum);
	}

	template <class T>
	Bounds3<T>::Bounds3(Point3<T> const& p)
	{
		pMin = p;
		pMax = p;
	}

	template <class T>
	Bounds3<T>::Bounds3(Point3<T> const& p1, Point3<T> const& p2)
	{
		pMin = Vector3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z));
		pMax = Vector3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z));
	}

	template <class T>
	auto Bounds3<T>::operator[](uint32_t i) const -> Point3<T> const&
	{
		return (i == 0) ? pMin : pMax;
	}

	template <class T>
	auto Bounds3<T>::operator[](uint32_t i) -> Point3<T>&
	{
		return (i == 0) ? pMin : pMax;
	}

	template <class T>
	auto Bounds3<T>::corner(uint32_t c) const -> Point3<T>
	{
		return Point3<T>(
			(*this)[(c & 1)].x,
			(*this)[(c & 2) ? 1 : 0].y,
			(*this)[(c & 4) ? 1 : 0].z
			);
	}

	template <class T>
	auto Bounds3<T>::diagonal() const -> Vector3<T>
	{
		return pMax - pMin;
	}

	template <class T>
	auto Bounds3<T>::surfaceArea() const->T
	{
		Vector3<T> d = diagonal();
		return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
	}

	template <class T>
	auto Bounds3<T>::volume() const->T
	{
		Vector3<T> d = diagonal();
		return d.x * d.y * d.z;
	}

	template <class T>
	auto Bounds3<T>::maximumExtent() const-> uint32_t
	{
		Vector3<T> d = diagonal();
		if (d.x > d.y && d.x > d.z)
			return 0;
		else if (d.y > d.z)
			return 1;
		else
			return 2;
	}

	template <class T>
	auto Bounds3<T>::lerp(Point3<T> const& t) const -> Point3<T>
	{
		return Point3<T>({
			std::lerp(pMin.x, pMax.x, t.x),
			std::lerp(pMin.y, pMax.y, t.y),
			std::lerp(pMin.z, pMax.z, t.z)
			});
	}

	template <class T>
	auto Bounds3<T>::offset(Point3<T> const& p) const -> Vector3<T>
	{
		Vector3<T> o = p - pMin;
		if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
		if (pMax.y > pMin.y) o.x /= pMax.y - pMin.y;
		if (pMax.z > pMin.z) o.x /= pMax.z - pMin.z;
		return o;
	}
	
	template <class T>
	auto Bounds3<T>::boundingSphere(Point3<T>* center, float* radius) const -> void
	{
		*center = (pMin + pMax) / 2;
		*radius = inside(*center, *this) ? distance(*center, pMax) : 0;
	}

	template <class T>
	auto Bounds3<T>::intersectP(ray3 const& ray, float* hitt0, float* hitt1) const -> bool
	{
		float t0 = 0, t1 = ray.tMax;
		for (int i = 0; i < 3; ++i) {
			// could handle infinite (1/0) cases
			float invRayDir = 1 / ray.d[i];
			float tNear = (pMin[i] - ray.o[i]) * invRayDir;
			float tFar = (pMax[i] - ray.o[i]) * invRayDir;
			if (tNear > tFar) std::swap(tNear, tFar);
			// could handle NaN (0/0) cases
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar < t1 ? tFar : t1;
			if (t0 > t1) return false;
		}
		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}

	template <class T>
	auto Bounds3<T>::intersectP(ray3 const& ray, vec3 const& invDir, ivec3 dirIsNeg) const -> bool
	{
		Bounds3<float> const& bounds = *this;
		// Check for ray intersection against x and y slabs
		float tMin	= (bounds[0 + dirIsNeg[0]].x - ray.o.x) * invDir.x;
		float tMax	= (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
		float tyMin = (bounds[0 + dirIsNeg[1]].y - ray.o.y) * invDir.y;
		float tyMax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;
		if (tMin > tyMax || tyMin > tMax)
			return false;
		if (tyMin > tMin) tMin = tyMin;
		if (tyMax < tMax) tMax = tyMax;
		// Check for ray intersection against z slab
		float tzMin = (bounds[0 + dirIsNeg[2]].z - ray.o.z) * invDir.z;
		float tzMax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;
		if (tMin > tzMax || tzMin > tMax)
			return false;
		if (tzMin > tMin) tMin = tzMin;
		if (tzMax < tMax) tMax = tzMax;
		return (tMin < ray.tMax) && (tMax > 0);
	}

	export template <class T>
	inline auto unionPoint(Bounds3<T> const& b, Point3<T> const& p) noexcept -> Bounds3<T>
	{
		return Bounds3<T>(Point3<T>(std::min(b.pMin.x, p.x), 
									std::min(b.pMin.y, p.y),
									std::min(b.pMin.z, p.z)),
						  Point3<T>(std::max(b.pMax.x, p.x),
									std::max(b.pMax.y, p.y),
									std::max(b.pMax.z, p.z)));
	}

	export template <class T>
	inline auto unionBounds(Bounds3<T> const& b1, Bounds3<T> const& b2) noexcept -> Bounds3<T>
	{
		return Bounds3<T>(Point3<T>(std::min(b1.pMin.x, b2.pMin.x),
									std::min(b1.pMin.y, b2.pMin.y),
									std::min(b1.pMin.z, b2.pMin.z)),
						  Point3<T>(std::max(b1.pMax.x, b2.pMax.x),
									std::max(b1.pMax.y, b2.pMax.y),
									std::max(b1.pMax.z, b2.pMax.z)));
	}

	export template <class T>
	inline auto intersect(Bounds3<T> const& b1, Bounds3<T> const& b2) noexcept -> Bounds3<T>
	{
		return Bounds3<T>(Point3<T>(std::max(b1.pMin.x, b2.pMin.x),
									std::max(b1.pMin.y, b2.pMin.y),
									std::max(b1.pMin.z, b2.pMin.z)),
						  Point3<T>(std::min(b1.pMax.x, b2.pMax.x),
									std::min(b1.pMax.y, b2.pMax.y),
									std::min(b1.pMax.z, b2.pMax.z)));
	}

	export template <class T>
	inline auto overlaps(Bounds3<T> const& b1, Bounds3<T> const& b2) noexcept -> Bounds3<T>
	{
		bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
		bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
		bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
		return (x && y && z);
	}

	export template <class T>
	inline auto inside(Point3<T> const& p, Bounds3<T> const& b) noexcept -> bool
	{
		return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
				p.y >= b.pMin.y && p.y <= b.pMax.y &&
				p.z >= b.pMin.z && p.z <= b.pMax.z);
	}

	export template <class T>
	inline auto insideExclusive(Point3<T> const& p, Bounds3<T> const& b) noexcept -> bool
	{
		return (p.x >= b.pMin.x && p.x < b.pMax.x &&
				p.y >= b.pMin.y && p.y < b.pMax.y &&
				p.z >= b.pMin.z && p.z < b.pMax.z);
	}

	export template <class T>
	inline auto expand(Bounds3<T> const& b, float delta) noexcept -> bool
	{
		return Bounds3<T>(b.pMin - Vector3<T>(delta),
						  b.pMax + Vector3<T>(delta));
	}

	export using bounds3 = Bounds3<float>;
	export using ibounds3 = Bounds3<int32_t>;
}