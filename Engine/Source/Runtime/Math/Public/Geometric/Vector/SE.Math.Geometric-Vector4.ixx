module;
#include <cstdint>
#include <cstdlib>
#include <cmath>
export module SE.Math.Geometric:Vector4;
import :Vector2;
import :Vector3;

namespace SIByL::Math
{
	export template <class T>
	struct Vector4
	{
		// __declspec(align(16))
		union {
			T data[4];
			struct { T x, y, z, w; };
			struct { T r, g, b, a; };
			struct { T s, t, p, q; };
		};

		auto lengthSquared() const -> float;
		auto length() const -> float;

		template <class U>
		explicit operator Vector4<U>() const;

		Vector4() :x(0), y(0), z(0), w(0) {}
		Vector4(T const& _v) :x(_v), y(_v), z(_v), w(_v) {}
		Vector4(Vector2<T> const& _v) :x(_v.x), y(_v.y), z(0), w(0) {}
		Vector4(Vector3<T> const& _v) :x(_v.x), y(_v.y), z(_v.z), w(0) {}
		Vector4(T const& _x, T const& _y, T const& _z = 0, T const& _w = 0) :x(_x), y(_y), z(_z), w(_w) {}

		operator T* () { return data; }
		operator const T* const () { return static_cast<const T*>(data); }
		auto operator [](size_t idx) ->T& { return data[idx]; }
		auto operator [](size_t idx) const ->T const& { return data[idx]; }

		auto operator-() const->Vector4<T>;
		auto operator*(T s) const->Vector4<T>;
		auto operator/(T s) const->Vector4<T>;
		auto operator*=(T s) const->Vector4<T>&;
		auto operator/=(T s) const->Vector4<T>&;
		auto operator+(Vector4<T> const& v) const->Vector4<T>;
		auto operator-(Vector4<T> const& v) const->Vector4<T>;
		auto operator*(Vector4<T> const& v) const->Vector4<T>;
		auto operator+=(Vector4<T> const& v)->Vector4<T>&;
		auto operator-=(Vector4<T> const& v)->Vector4<T>&;
		auto operator*=(Vector4<T> const& v)->Vector4<T>&;
		auto operator==(Vector4<T> const& v) const -> bool;
		auto operator!=(Vector4<T> const& v) const -> bool;
	};

	export using vec4 = Vector4<float>;
	export using ivec4 = Vector4<int32_t>;
	export using uvec4 = Vector4<uint32_t>;

	template <class T>
	template <class U>
	Vector4<T>::operator Vector4<U>() const
	{
		return Vector4<U>(x, y, z, w);
	}

	export template <class S, class T> auto operator*(S s, Vector4<T> const& v) noexcept -> Vector4<T> {
		return Vector4<T>{v.x* s, v.y* s, v.z* s, v.w* s};
	}

	export template <class T> inline auto abs(Vector4<T> const& v) noexcept -> T
	{
		T result;
		for (size_t i = 0; i < 4; ++i) {
			result = std::abs(v.data[i]);
		}
		return result;
	}

	export template <class T> inline auto floor(Vector4<T> const& v) noexcept -> Vector4<T>
	{
		Vector4<T> result;
		for (size_t i = 0; i < 4; ++i) {
			result.data[i] = std::floor(v.data[i]);
		}
		return result;
	}
	
	export template <class T> inline auto ceil(Vector4<T> const& v) noexcept -> Vector4<T>
	{
		Vector4<T> result;
		for (size_t i = 0; i < 4; ++i) {
			result.data[i] = std::ceil(v.data[i]);
		}
		return result;
	}

	export template <class T> inline auto dot(Vector4<T> const& x, Vector4<T> const& y) noexcept -> T
	{
		T result;
		for (size_t i = 0; i < 4; ++i) {
			result += x[i] * y[i];
		}
		return result;
	}

	export template <class T> inline auto absDot(Vector4<T> const& x, Vector4<T> const& y) noexcept -> T
	{
		return std::abs(dot(x, y));
	}

	export template <class T> inline auto normalize(Vector4<T> const& v) noexcept -> Vector4<T>
	{
		return v / v.length();
	}

	export template <class T> inline auto minComponent(Vector4<T> const& v) noexcept -> T
	{
		return std::min(std::min(v.x, v.y), std::min(v.z, v.w));
	}

	export template <class T> inline auto maxComponent(Vector4<T> const& v) noexcept -> T
	{
		return std::max(std::max(v.x, v.y), std::max(v.z, v.w));
	}

	export template <class T> inline auto maxDimension(Vector4<T> const& v) noexcept -> size_t
	{
		return (v.x > v.y) ? ((v.x > v.z) ? ((v.x > v.w) ? 0 : 3) : ((v.z > v.w) ? 2 : 3)) : 
			((v.y > v.z) ? ((v.y > v.w) ? 1 : 3) : ((v.z > v.w) ? 2 : 3));
	}

	export template <class T> inline auto minDimension(Vector4<T> const& v) noexcept -> size_t
	{
		return (v.x < v.y) ? ((v.x < v.z) ? ((v.x < v.w) ? 0 : 3) : ((v.z < v.w) ? 2 : 3)) :
			((v.y < v.z) ? ((v.y < v.w) ? 1 : 3) : ((v.z < v.w) ? 2 : 3));
	}

	export template <class T> inline auto max(Vector4<T> const& x, Vector4<T> const& y) noexcept -> Vector4<T>
	{
		Vector4<T> result;
		for (size_t i = 0; i < 4; ++i) {
			result[i] = std::max(x[i], y[i]);
		}
		return result;
	}

	export template <class T> inline auto min(Vector4<T> const& x, Vector4<T> const& y) noexcept -> Vector4<T>
	{
		Vector4<T> result;
		for (size_t i = 0; i < 4; ++i) {
			result[i] = std::min(x[i], y[i]);
		}
		return result;
	}

	export template <class T> inline auto permute(Vector4<T> const& v, size_t x, size_t y, size_t z, size_t w) noexcept -> Vector4<T>
	{
		return Vector4<T>(v[x], v[y], v[z], v[w]);
	}

	export template <class T> inline auto distance(Vector4<T> const& p1, Vector4<T> const& p2) noexcept -> float
	{
		return (p1 - p2).length();
	}

	export template <class T> inline auto distanceSquared(Vector4<T> const& p1, Vector4<T> const& p2) noexcept -> float
	{
		return (p1 - p2).lengthSquared();
	}

	export template <class T> inline auto operator*(T s, Vector4<T>& v) -> Vector4<T>
	{
		return v * s;
	}

	export template <class T> inline auto lerp(float t, Vector4<T> const& x, Vector4<T>& y) noexcept -> Vector4<T>
	{
		return (1 - t) * x + t * y;
	}

	template <class T>
	auto Vector4<T>::lengthSquared() const -> float
	{
		return x * x + y * y + z * z + w * w;
	}

	template <class T>
	auto Vector4<T>::length() const -> float
	{
		return std::sqrt(lengthSquared());
	}

	template <class T>
	auto Vector4<T>::operator-() const->Vector4<T>
	{
		return Vector4<T>{-x, -y, -z, -w};
	}

	template <class T>
	auto Vector4<T>::operator*(T s) const->Vector4<T>
	{
		Vector4<T> result;
		for (size_t i = 0; i < 4; i++) {
			result.data[i] = data[i] * s;
		}
		return result;
	}
	
	template <class T>
	auto Vector4<T>::operator/(T s) const->Vector4<T>
	{
		float inv = 1.f / s;
		Vector4<T> result;
		for (size_t i = 0; i < 4; i++) {
			result.data[i] = data[i] * inv;
		}
		return result;
	}

	template <class T>
	auto Vector4<T>::operator*=(T s) const->Vector4<T>&
	{
		for (size_t i = 0; i < 4; i++) {
			data[i] *= s;
		}
		return *this;
	}
	
	template <class T>
	auto Vector4<T>::operator/=(T s) const->Vector4<T>&
	{
		float inv = 1.f / s;
		for (size_t i = 0; i < 4; i++) {
			data[i] *= inv;
		}
		return *this;
	}

	template <class T>
	auto Vector4<T>::operator+(Vector4<T> const& v) const->Vector4<T>
	{
		Vector4<T> result;
		for (size_t i = 0; i < 4; i++) {
			result.data[i] = data[i] + v.data[i];
		}
		return result;
	}

	template <class T>
	auto Vector4<T>::operator-(Vector4<T> const& v) const->Vector4<T>
	{
		Vector4<T> result;
		for (size_t i = 0; i < 4; i++) {
			result.data[i] = data[i] - v.data[i];
		}
		return result;
	}

	template <class T>
	auto Vector4<T>::operator*(Vector4<T> const& v) const->Vector4<T>
	{
		Vector4<T> result;
		for (size_t i = 0; i < 4; i++) {
			result.data[i] = data[i] * v.data[i];
		}
		return result;
	}

	template <class T>
	auto Vector4<T>::operator+=(Vector4<T> const& v)->Vector4<T>&
	{
		for (size_t i = 0; i < 4; i++) {
			data[i] += v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector4<T>::operator-=(Vector4<T> const& v)->Vector4<T>&
	{
		for (size_t i = 0; i < 4; i++) {
			data[i] -= v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector4<T>::operator*=(Vector4<T> const& v)->Vector4<T>&
	{
		for (size_t i = 0; i < 4; i++) {
			data[i] *= v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector4<T>::operator==(Vector4<T> const& v) const -> bool
	{
		return (x == v.x) && (y == v.y) && (z == v.z) && (w == v.w);
	}

	template <class T>
	auto Vector4<T>::operator!=(Vector4<T> const& v) const -> bool
	{
		return !(*this == v);
	}
}