module;
#include <cstdint>
export module Math.Geometry:Normal3;
import Math.Vector;

namespace SIByL::Math
{
	export template <class T>
		struct Normal3 :public Vector3<T>
	{
		Normal3(T const& _x, T const& _y, T const& _z = 0)
			:Vector3<T>(_x, _y, _z) {}

		Normal3(Vector3<T> const& v = { 0,0,0 })
			:Vector3<T>(v) {}

		template <typename U>
		explicit Normal3(Normal3<U> const& p)
			:Vector3<T>((T)p.x, (T)p.y, (T)p.z) {}

		template <typename U>
		explicit operator Vector3<U>() const { return Vector3<U>(this->x, this->y, this->z); }
	};

	export using normal3 = Normal3<float>;
	export using inormal3 = Normal3<int32_t>;
	export using unormal3 = Normal3<uint32_t>;
}