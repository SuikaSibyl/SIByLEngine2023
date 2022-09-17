module;
#include <cstdint>
#include <cstdlib>
#include <cmath>
export module Math.Vector:Vector3;
import :Vector2;

namespace SIByL::Math
{
	export template <class T>
	struct Vector3
	{
		union {
			T data[3];
			struct { T x, y, z; };
			struct { T r, g, b; };
			struct { T s, t, p; };
		};

		Vector3() :x(0), y(0), z(0) {}
		Vector3(T const& _v) :x(_v), y(_v), z(_v) {}
		Vector3(Vector2<T> const& _v) :x(_v.x), y(_v.y), z(0) {}
		Vector3(T const& _x, T const& _y, T const& _z = 0) :x(_x), y(_y), z(_z) {}

		auto lengthSquared() const -> float;
		auto length() const -> float;

		template <class U>
		explicit operator Vector3<U>() const;

		operator T* () { return data; }
		operator const T* const () { return static_cast<const T*>(data); }
		auto operator [](size_t idx) ->T& { return data[idx]; }
		auto operator [](size_t idx) const ->T const& { return data[idx]; }

		auto operator-() const->Vector3<T>;
		auto operator*(T s) const->Vector3<T>;
		auto operator/(T s) const->Vector3<T>;
		auto operator/=(T s) const->Vector3<T>&;
		auto operator+(Vector3<T> const& v) const->Vector3<T>;
		auto operator-(Vector3<T> const& v) const->Vector3<T>;
		auto operator*(Vector3<T> const& v) const->Vector3<T>;
		auto operator+=(Vector3<T> const& v)->Vector3<T>&;
		auto operator-=(Vector3<T> const& v)->Vector3<T>&;
		auto operator*=(Vector3<T> const& v)->Vector3<T>&;
		auto operator==(Vector3<T> const& v) const -> bool;
		auto operator!=(Vector3<T> const& v) const -> bool;
	};

	export using vec3 = Vector3<float>;
	export using ivec3 = Vector3<int32_t>;
	export using uvec3 = Vector3<uint32_t>;

	template <class T>
	template <class U>
	Vector3<T>::operator Vector3<U>() const
	{
		return Vector3<U>(x, y, z);
	}

	export template <class S, class T> auto operator*(S s, Vector3<T> const& v) noexcept -> Vector3<T> {
		return Vector3<T>{v.x* s, v.y* s, v.z* s};
	}

	export template <class T> inline auto abs(Vector3<T> const& v) noexcept -> T
	{
		T result;
		for (size_t i = 0; i < 3; ++i) {
			result = std::abs(v.data[i]);
		}
		return result;
	}

	export template <class T> inline auto floor(Vector3<T> const& v) noexcept -> T
	{
		T result;
		for (size_t i = 0; i < 3; ++i) {
			result = std::floor(v.data[i]);
		}
		return result;
	}

	export template <class T> inline auto ceil(Vector3<T> const& v) noexcept -> T
	{
		T result;
		for (size_t i = 0; i < 3; ++i) {
			result = std::ceil(v.data[i]);
		}
		return result;
	}

	export template <class T> inline auto dot(Vector3<T> const& x, Vector3<T> const& y) noexcept -> T
	{
		T result = 0;
		for (size_t i = 0; i < 3; ++i) {
			result += x[i] * y[i];
		}
		return result;
	}

	export template <class T> inline auto absDot(Vector3<T> const& x, Vector3<T> const& y) noexcept -> T
	{
		return std::abs(dot(x, y));
	}

	export template <class T> inline auto cross(Vector3<T> const& x, Vector3<T> const& y) noexcept -> Vector3<T>
	{
		return Vector3<T>(
			(x[1] * y[2]) - (x[2] * y[1]), 
			(x[2] * y[0]) - (x[0] * y[2]), 
			(x[0] * y[1]) - (x[1] * y[0]));
	}

	export template <class T> inline auto normalize(Vector3<T> const& v) noexcept -> Vector3<T>
	{
		return v / v.length();
	}

	export template <class T> inline auto minComponent(Vector3<T> const& v) noexcept -> T
	{
		return std::min(std::min(v.x, v.y), v.z);
	}

	export template <class T> inline auto maxComponent(Vector3<T> const& v) noexcept -> T
	{
		return std::max(std::max(v.x, v.y), v.z);
	}
	
	export template <class T> inline auto maxDimension(Vector3<T> const& v) noexcept -> size_t
	{
		return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
	}

	export template <class T> inline auto minDimension(Vector3<T> const& v) noexcept -> size_t
	{
		return (v.x < v.y) ? ((v.x < v.z) ? 0 : 2) : ((v.y < v.z) ? 1 : 2);
	}

	export template <class T> inline auto max(Vector3<T> const& x, Vector3<T>& y) noexcept -> Vector3<T>
	{
		Vector3<T> result;
		for (size_t i = 0; i < 3; ++i) {
			result[i] = std::max(x[i], y[i]);
		}
		return result;
	}

	export template <class T> inline auto min(Vector3<T> const& x, Vector3<T>& y) noexcept -> Vector3<T>
	{
		Vector3<T> result;
		for (size_t i = 0; i < 3; ++i) {
			result[i] = std::min(x[i], y[i]);
		}
		return result;
	}
	export template <class T> inline auto permute(Vector3<T> const& v, size_t x, size_t y, size_t z) noexcept -> Vector3<T>
	{
		return Vector3<T>(v[x], v[y], v[z]);
	}

	export template <class T> inline auto distance(Vector3<T> const& p1, Vector3<T> const& p2) noexcept -> float
	{
		return (p1 - p2).length();
	}

	export template <class T> inline auto distanceSquared(Vector3<T> const& p1, Vector3<T> const& p2) noexcept -> float
	{
		return (p1 - p2).lengthSquared();
	}

	export template <class T> inline auto operator*(T s, Vector3<T>& v) -> Vector3<T>
	{
		return v * s;
	}

	export template <class T> inline auto lerp(float t, Vector3<T> const& x, Vector3<T>& y) noexcept -> Vector3<T>
	{
		return (1 - t) * x + t * y;
	}

	export template <class T> inline auto faceForward(Vector3<T> const& n, Vector3<T> const& v) noexcept -> Vector3<T>
	{
		return (dot(n, v) < 0.0f) ? -n : n;
	}

	template <class T>
	auto Vector3<T>::lengthSquared() const -> float
	{
		return x * x + y * y + z * z;
	}

	template <class T>
	auto Vector3<T>::length() const -> float
	{
		return std::sqrt(lengthSquared());
	}

	template <class T>
	auto Vector3<T>::operator-() const->Vector3<T>
	{
		return Vector3<T>{-x, -y, -z};
	}

	template <class T>
	auto Vector3<T>::operator*(T s) const->Vector3<T>
	{
		Vector3<T> result;
		for (size_t i = 0; i < 3; i++) {
			result.data[i] = data[i] * s;
		}
		return result;
	}

	template <class T>
	auto Vector3<T>::operator/(T s) const->Vector3<T>
	{
		float inv = 1.f / s;
		Vector3<T> result;
		for (size_t i = 0; i < 3; i++) {
			result.data[i] = data[i] * inv;
		}
		return result;
	}

	template <class T>
	auto Vector3<T>::operator/=(T s) const->Vector3<T>&
	{
		float inv = 1.f / s;
		for (size_t i = 0; i < 3; i++) {
			data[i] *= inv;
		}
		return *this;
	}

	template <class T>
	auto Vector3<T>::operator+(Vector3<T> const& v) const->Vector3<T>
	{
		Vector3<T> result;
		for (size_t i = 0; i < 3; i++) {
			result.data[i] = data[i] + v.data[i];
		}
		return result;
	}

	template <class T>
	auto Vector3<T>::operator-(Vector3<T> const& v) const->Vector3<T>
	{
		Vector3<T> result;
		for (size_t i = 0; i < 3; i++) {
			result.data[i] = data[i] - v.data[i];
		}
		return result;
	}
	
	template <class T>
	auto Vector3<T>::operator*(Vector3<T> const& v) const->Vector3<T>
	{
		Vector3<T> result;
		for (size_t i = 0; i < 3; i++) {
			result.data[i] = data[i] * v.data[i];
		}
		return result;
	}

	template <class T>
	auto Vector3<T>::operator+=(Vector3<T> const& v) ->Vector3<T>&
	{
		for (size_t i = 0; i < 3; i++) {
			data[i] += v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector3<T>::operator-=(Vector3<T> const& v)->Vector3<T>&
	{
		for (size_t i = 0; i < 3; i++) {
			data[i] -= v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector3<T>::operator*=(Vector3<T> const& v)->Vector3<T>&
	{
		for (size_t i = 0; i < 3; i++) {
			data[i] *= v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector3<T>::operator==(Vector3<T> const& v) const -> bool
	{
		return (x == v.x) && (y == v.y) && (z == v.z);
	}

	template <class T>
	auto Vector3<T>::operator!=(Vector3<T> const& v) const -> bool
	{
		return !(*this == v);
	}
}