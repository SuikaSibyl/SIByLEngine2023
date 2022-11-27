module;
#include <cstdint>
#include <cstdlib>
#include <cmath>
export module Math.Vector:Vector2;

namespace SIByL::Math
{
	export template <class T>
	struct Vector2
	{
		union {
			T data[2];
			struct { T x, y; };
			struct { T s, g; };
			struct { T u, v; };
			struct { T s, t; };
		};

		Vector2() :x(0), y(0) {}
		Vector2(T const& _v) :x(_v), y(_v) {}
		Vector2(T const& _x, T const& _y) :x(_x), y(_y) {}

		auto lengthSquared() const -> float;
		auto length() const -> float;

		template <class U>
		explicit operator Vector2<U>() const;

		operator T* () { return data; }
		operator const T* const () { return static_cast<const T*>(data); }
		auto operator [](size_t idx) ->T& { return data[idx]; }
		auto operator [](size_t idx) const ->T const& { return data[idx]; }

		auto operator-() const ->Vector2<T>;
		auto operator*(T s) const->Vector2<T>;
		auto operator/(T s) const->Vector2<T>;
		auto operator*=(T s) const->Vector2<T>&;
		auto operator/=(T s) const->Vector2<T>&;
		auto operator+(Vector2<T> const& v) const -> Vector2<T>;
		auto operator-(Vector2<T> const& v) const -> Vector2<T>;
		auto operator*(Vector2<T> const& v) const -> Vector2<T>;
		auto operator+=(Vector2<T> const& v) const->Vector2<T>&;
		auto operator-=(Vector2<T> const& v) const->Vector2<T>&;
		auto operator*=(Vector2<T> const& v) const->Vector2<T>&;
		auto operator==(Vector2<T> const& v) const -> bool;
		auto operator!=(Vector2<T> const& v) const -> bool;
	};
	
	export using vec2 = Vector2<float>;
	export using ivec2 = Vector2<int32_t>;
	export using uvec2 = Vector2<uint32_t>;
	export using dvec2 = Vector2<double>;

	template <class T>
	template <class U>
	Vector2<T>::operator Vector2<U>() const
	{
		return Vector2<U>(x, y);
	}

	export template <class S, class T> auto operator*(S s, Vector2<T> const& v) noexcept -> Vector2<T> {
		return Vector2<T>{v.x* s, v.y* s};
	}

	export template <class T> inline auto abs(Vector2<T> const& v) noexcept -> T
	{
		T result;
		for (size_t i = 0; i < 2; ++i) {
			result = std::abs(v.data[i]);
		}
		return result;
	}

	export template <class T> inline auto floor(Vector2<T> const& v) noexcept -> Vector2<T>
	{
		Vector2<T> result;
		for (size_t i = 0; i < 2; ++i) {
			result.data[i] = std::ceil(v.data[i]);
		}
		return result;
	}

	export template <class T> inline auto ceil(Vector2<T> const& v) noexcept -> Vector2<T>
	{
		Vector2<T> result;
		for (size_t i = 0; i < 2; ++i) {
			result.data[i] = std::ceil(v.data[i]);
		}
		return result;
	}

	export template <class T> inline auto dot(Vector2<T> const& x, Vector2<T>& y) noexcept -> T
	{
		T result;
		for (size_t i = 0; i < 2; ++i) {
			result += x[i] * y[i];
		}
		return result;
	}

	export template <class T> inline auto absDot(Vector2<T> const& x, Vector2<T>& y) noexcept -> T
	{
		return std::abs(dot(x, y));
	}

	export template <class T> inline auto cross(Vector2<T> const& x, Vector2<T>& y) noexcept -> T
	{
		return x[0] * y[1] - x[1] * y[0];
	}

	export template <class T> inline auto normalize(Vector2<T> const& v) noexcept -> Vector2<T>
	{
		return v / v.length();
	}

	export template <class T> inline auto minComponent(Vector2<T> const& v) noexcept -> T
	{
		return std::min(v.x, v.y);
	}

	export template <class T> inline auto maxComponent(Vector2<T> const& v) noexcept -> T
	{
		return std::max(v.x, v.y);
	}

	export template <class T> inline auto maxDimension(Vector2<T> const& v) noexcept -> size_t
	{
		return (v.x > v.y) ? 0 : 1;
	}

	export template <class T> inline auto minDimension(Vector2<T> const& v) noexcept -> size_t
	{
		return (v.x < v.y) ? 0 : 1;
	}

	export template <class T> inline auto max(Vector2<T> const& x, Vector2<T>& y) noexcept -> Vector2<T>
	{
		Vector2<T> result;
		for (size_t i = 0; i < 2; ++i) {
			result[i] = std::max(x[i], y[i]);
		}
		return result;
	}

	export template <class T> inline auto min(Vector2<T> const& x, Vector2<T>& y) noexcept -> Vector2<T>
	{
		Vector2<T> result;
		for (size_t i = 0; i < 2; ++i) {
			result[i] = std::min(x[i], y[i]);
		}
		return result;
	}

	export template <class T> inline auto permute(Vector2<T> const& v, size_t x, size_t y) noexcept -> Vector2<T>
	{
		return Vector2<T>(v[x], v[y]);
	}

	export template <class T> inline auto distance(Vector2<T> const& p1, Vector2<T> const& p2) noexcept -> float
	{
		return (p1 - p2).length();
	}

	export template <class T> inline auto distanceSquared(Vector2<T> const& p1, Vector2<T> const& p2) noexcept -> float
	{
		return (p1 - p2).lengthSquared();
	}

	export template <class T> inline auto operator*(T s, Vector2<T>& v) -> Vector2<T>
	{
		return v * s;
	}

	export template <class T> inline auto lerp(float t, Vector2<T> const& x, Vector2<T>& y) noexcept -> Vector2<T>
	{
		return (1 - t) * x + t * y;
	}

	template <class T>
	auto Vector2<T>::lengthSquared() const -> float
	{
		return x * x + y * y;
	}

	template <class T>
	auto Vector2<T>::length() const -> float
	{
		return std::sqrt(lengthSquared());
	}

	template <class T>
	auto Vector2<T>::operator-() const->Vector2<T>
	{
		return Vector2<T>{-x, -y};
	}

	template <class T>
	auto Vector2<T>::operator*(T s) const->Vector2<T>
	{
		Vector2<T> result;
		for (size_t i = 0; i < 2; i++) {
			result.data[i] = data[i] * s;
		}
		return result;
	}

	template <class T>
	auto Vector2<T>::operator/(T s) const->Vector2<T>
	{
		float inv = 1.f / s;
		Vector2<T> result;
		for (size_t i = 0; i < 2; i++) {
			result.data[i] = data[i] * inv;
		}
		return result;
	}

	template <class T>
	auto Vector2<T>::operator*=(T s) const->Vector2<T>&
	{
		for (size_t i = 0; i < 2; i++) {
			data[i] *= s;
		}
		return *this;
	}

	template <class T>
	auto Vector2<T>::operator/=(T s) const->Vector2<T>&
	{
		float inv = 1.f / s;
		for (size_t i = 0; i < 2; i++) {
			data[i] *= inv;
		}
		return *this;
	}

	template <class T>
	auto Vector2<T>::operator+(Vector2<T> const& v) const->Vector2<T>
	{
		Vector2<T> result;
		for (size_t i = 0; i < 2; i++) {
			result.data[i] = data[i] + v.data[i];
		}
		return result;
	}

	template <class T>
	auto Vector2<T>::operator-(Vector2<T> const& v) const->Vector2<T>
	{
		Vector2<T> result;
		for (size_t i = 0; i < 2; i++) {
			result.data[i] = data[i] - v.data[i];
		}
		return result;
	}

	template <class T>
	auto Vector2<T>::operator*(Vector2<T> const& v) const->Vector2<T>
	{
		Vector2<T> result;
		for (size_t i = 0; i < 2; i++) {
			result.data[i] = data[i] * v.data[i];
		}
		return result;
	}

	template <class T>
	auto Vector2<T>::operator+=(Vector2<T> const& v) const->Vector2<T>&
	{
		for (size_t i = 0; i < 2; i++) {
			data[i] += v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector2<T>::operator-=(Vector2<T> const& v) const->Vector2<T>&
	{
		for (size_t i = 0; i < 2; i++) {
			data[i] -= v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector2<T>::operator*=(Vector2<T> const& v) const->Vector2<T>&
	{
		for (size_t i = 0; i < 2; i++) {
			data[i] *= v.data[i];
		}
		return *this;
	}

	template <class T>
	auto Vector2<T>::operator==(Vector2<T> const& v) const -> bool
	{
		return (x == v.x) && (y == v.y);
	}

	template <class T>
	auto Vector2<T>::operator!=(Vector2<T> const& v) const -> bool
	{
		return !(*this == v);
	}
}