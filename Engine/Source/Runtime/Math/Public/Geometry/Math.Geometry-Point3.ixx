module;
#include <cstdint>
export module Math.Geometry:Point3;
import Math.Vector;

namespace SIByL::Math
{
	export template <class T>
		struct Point3 :public Vector3<T>
	{
		Point3(T const& _x, T const& _y, T const& _z = 0)
			:Vector3<T>(_x, _y, _z) {}

		Point3(Vector3<T> const& v = { 0,0,0 })
			:Vector3<T>(v) {}

		template<class U>
		Point3(Vector3<U> const& v = { 0,0,0 })
			:Vector3<T>((T)v.x, (T)v.y, (T)v.z) {}

		template <typename U>
		explicit Point3(Point3<U> const& p)
			:Vector3<T>((T)p.x, (T)p.y, (T)p.z) {}

		template <typename U>
		explicit operator Vector3<U>() const { return Vector3<U>(this->x, this->y, this->z); }
	};

	export using point3 = Point3<float>;
	export using ipoint3 = Point3<int32_t>;
	export using upoint3 = Point3<uint32_t>;
}