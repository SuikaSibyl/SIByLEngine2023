#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <pmmintrin.h>
#include <xmmintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iterator>
#include <algorithm>

#include "SE.Math.Misc.hpp"

namespace SIByL::Math {

SE_EXPORT template <class T>
struct Vector2 {
  union {
    T data[2];
    struct {
      T x, y;
    };
    struct {
      T r, g;
    };
    struct {
      T u, v;
    };
    struct {
      T s, t;
    };
  };

  Vector2() : x(0), y(0) {}
  Vector2(T const& _v) : x(_v), y(_v) {}
  Vector2(T const& _x, T const& _y) : x(_x), y(_y) {}

  auto lengthSquared() const -> float;
  auto length() const -> float;

  template <class U>
  explicit operator Vector2<U>() const;

  operator T*() { return data; }
  operator const T* const() { return static_cast<const T*>(data); }
  auto operator[](size_t idx) -> T& { return data[idx]; }
  auto operator[](size_t idx) const -> T const& { return data[idx]; }

  auto operator-() const -> Vector2<T>;
  auto operator*(T s) const -> Vector2<T>;
  auto operator/(T s) const -> Vector2<T>;
  auto operator+(Vector2<T> const& v) const -> Vector2<T>;
  auto operator-(Vector2<T> const& v) const -> Vector2<T>;
  auto operator*(Vector2<T> const& v) const -> Vector2<T>;
  auto operator/(Vector2<T> const& v) const -> Vector2<T>;
  auto operator==(Vector2<T> const& v) const -> bool;
  auto operator!=(Vector2<T> const& v) const -> bool;
  auto operator*=(T s) -> Vector2<T>&;
  auto operator/=(T s) -> Vector2<T>&;
  auto operator+=(Vector2<T> const& v) -> Vector2<T>&;
  auto operator-=(Vector2<T> const& v) -> Vector2<T>&;
  auto operator*=(Vector2<T> const& v) -> Vector2<T>&;
  auto operator/=(Vector2<T> const& v) -> Vector2<T>&;
};

SE_EXPORT using bvec2 = Vector2<bool>;
SE_EXPORT using vec2 = Vector2<float>;
SE_EXPORT using ivec2 = Vector2<int32_t>;
SE_EXPORT using uvec2 = Vector2<uint32_t>;
SE_EXPORT using dvec2 = Vector2<double>;
SE_EXPORT using svec2 = Vector2<size_t>;

template <class T>
template <class U>
Vector2<T>::operator Vector2<U>() const {
  return Vector2<U>(x, y);
}

SE_EXPORT template <class S, class T>
auto operator*(S s, Vector2<T> const& v) noexcept -> Vector2<T> {
  return Vector2<T>{v.x * s, v.y * s};
}

SE_EXPORT template <class T>
inline auto abs(Vector2<T> const& v) noexcept -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; ++i) {
    result.data[i] = std::abs(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto floor(Vector2<T> const& v) noexcept -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; ++i) {
    result.data[i] = std::floor(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto ceil(Vector2<T> const& v) noexcept -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; ++i) {
    result.data[i] = std::ceil(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto dot(Vector2<T> const& x, Vector2<T> const& y) noexcept -> T {
  T result = 0;
  for (size_t i = 0; i < 2; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

SE_EXPORT template <class T>
inline auto absDot(Vector2<T> const& x, Vector2<T> const& y) noexcept -> T {
  return std::abs(dot(x, y));
}

SE_EXPORT template <class T>
inline auto cross(Vector2<T> const& x, Vector2<T> const& y) noexcept -> T {
  return x[0] * y[1] - x[1] * y[0];
}

SE_EXPORT template <class T>
inline auto normalize(Vector2<T> const& v) noexcept -> Vector2<T> {
  return v / v.length();
}

SE_EXPORT template <class T>
inline auto length(Vector2<T> const& v) noexcept -> T {
  return v.length();
}

SE_EXPORT inline auto sign(float x) noexcept -> float {
  if (x > 0) return 1.f;
  else if (x<0) return -1.f;
  else return 0.f;
}

SE_EXPORT template <class T>
inline auto sign(Vector2<T> const& v) noexcept -> Vector2<float> {
  return Vector2<float> { sign(v.x), sign(v.y) };
}

SE_EXPORT template <class T>
inline auto equal(Vector2<T> const& v1, Vector2<T> const& v2) noexcept -> bool {
  return v1.x == v2.x && v1.y == v2.y;
}

SE_EXPORT template <class T>
inline auto minComponent(Vector2<T> const& v) noexcept -> T {
  return std::min(v.x, v.y);
}

SE_EXPORT template <class T>
inline auto maxComponent(Vector2<T> const& v) noexcept -> T {
  return std::max(v.x, v.y);
}

SE_EXPORT template <class T>
inline auto maxDimension(Vector2<T> const& v) noexcept -> size_t {
  return (v.x > v.y) ? 0 : 1;
}

SE_EXPORT template <class T>
inline auto minDimension(Vector2<T> const& v) noexcept -> size_t {
  return (v.x < v.y) ? 0 : 1;
}

SE_EXPORT template <class T>
inline auto max(Vector2<T> const& x, Vector2<T> const& y) noexcept
    -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; ++i) {
    result[i] = std::max(x[i], y[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto min(Vector2<T> const& x, Vector2<T> const& y) noexcept
    -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; ++i) {
    result[i] = std::min(x[i], y[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto permute(Vector2<T> const& v, size_t x, size_t y) noexcept
    -> Vector2<T> {
  return Vector2<T>(v[x], v[y]);
}

SE_EXPORT template <class T>
inline auto distance(Vector2<T> const& p1, Vector2<T> const& p2) noexcept
    -> float {
  return (p1 - p2).length();
}

SE_EXPORT template <class T>
inline auto distanceSquared(Vector2<T> const& p1, Vector2<T> const& p2) noexcept
    -> float {
  return (p1 - p2).lengthSquared();
}

SE_EXPORT template <class T>
inline auto operator*(T s, Vector2<T>& v) -> Vector2<T> {
  return v * s;
}

SE_EXPORT template <class T>
inline auto lerp(float t, Vector2<T> const& x, Vector2<T>& y) noexcept
    -> Vector2<T> {
  return (1 - t) * x + t * y;
}

template <class T>
auto Vector2<T>::lengthSquared() const -> float {
  return x * x + y * y;
}

template <class T>
auto Vector2<T>::length() const -> float {
  return std::sqrt(lengthSquared());
}

template <class T>
auto Vector2<T>::operator-() const -> Vector2<T> {
  return Vector2<T>{-x, -y};
}

template <class T>
auto Vector2<T>::operator*(T s) const -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; i++) {
    result.data[i] = data[i] * s;
  }
  return result;
}

template <class T>
auto Vector2<T>::operator/(T s) const -> Vector2<T> {
  float inv = 1.f / s;
  Vector2<T> result;
  for (size_t i = 0; i < 2; i++) {
    result.data[i] = data[i] * inv;
  }
  return result;
}

template <class T>
auto Vector2<T>::operator*=(T s) -> Vector2<T>& {
  for (size_t i = 0; i < 2; i++) {
    data[i] *= s;
  }
  return *this;
}

template <class T>
auto Vector2<T>::operator/=(T s) -> Vector2<T>& {
  float inv = 1.f / s;
  for (size_t i = 0; i < 2; i++) {
    data[i] *= inv;
  }
  return *this;
}

template <class T>
auto Vector2<T>::operator+(Vector2<T> const& v) const -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; i++) {
    result.data[i] = data[i] + v.data[i];
  }
  return result;
}

template <class T>
auto Vector2<T>::operator-(Vector2<T> const& v) const -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; i++) {
    result.data[i] = data[i] - v.data[i];
  }
  return result;
}

template <class T>
auto Vector2<T>::operator*(Vector2<T> const& v) const -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; i++) {
    result.data[i] = data[i] * v.data[i];
  }
  return result;
}

template <class T>
auto Vector2<T>::operator/(Vector2<T> const& v) const -> Vector2<T> {
  Vector2<T> result;
  for (size_t i = 0; i < 2; i++) {
    result.data[i] = data[i] / v[i];
  }
  return result;
}

template <class T>
auto Vector2<T>::operator+=(Vector2<T> const& v) -> Vector2<T>& {
  for (size_t i = 0; i < 2; i++) {
    data[i] += v.data[i];
  }
  return *this;
}

template <class T>
auto Vector2<T>::operator-=(Vector2<T> const& v) -> Vector2<T>& {
  for (size_t i = 0; i < 2; i++) {
    data[i] -= v.data[i];
  }
  return *this;
}

template <class T>
auto Vector2<T>::operator*=(Vector2<T> const& v) -> Vector2<T>& {
  for (size_t i = 0; i < 2; i++) {
    data[i] *= v.data[i];
  }
  return *this;
}

template <class T>
auto Vector2<T>::operator/=(Vector2<T> const& v) -> Vector2<T>& {
  for (size_t i = 0; i < 2; i++) {
    data[i] /= v.data[i];
  }
  return *this;
}

template <class T>
auto Vector2<T>::operator==(Vector2<T> const& v) const -> bool {
  return (x == v.x) && (y == v.y);
}

template <class T>
auto Vector2<T>::operator!=(Vector2<T> const& v) const -> bool {
  return !(*this == v);
}

template <class T>
auto operator>=(Vector2<T> const& v1, Vector2<T> const& v2) -> Vector2<bool> {
  Vector2<bool> res;
  for (size_t i = 0; i < 2; i++) {
    res.data[i] = v1.data[i] >= v2.data[i];
  }
  return res;
}
template <class T>
auto operator>(Vector2<T> const& v1, Vector2<T> const& v2) -> Vector2<bool> {
  Vector2<bool> res;
  for (size_t i = 0; i < 2; i++) {
    res.data[i] = v1.data[i] > v2.data[i];
  }
  return res;
}
template <class T>
auto operator<=(Vector2<T> const& v1, Vector2<T> const& v2) -> Vector2<bool> {
  Vector2<bool> res;
  for (size_t i = 0; i < 2; i++) {
    res.data[i] = v1.data[i] <= v2.data[i];
  }
  return res;
}
template <class T>
auto operator<(Vector2<T> const& v1, Vector2<T> const& v2) -> Vector2<bool> {
  Vector2<bool> res;
  for (size_t i = 0; i < 2; i++) {
    res.data[i] = v1.data[i] < v2.data[i];
  }
  return res;
}
template <class T>
auto operator==(Vector2<T> const& v1, Vector2<T> const& v2) -> Vector2<bool> {
  Vector2<bool> res;
  for (size_t i = 0; i < 2; i++) {
    res.data[i] = v1.data[i] == v2.data[i];
  }
  return res;
}
template <class T>
auto operator!=(Vector2<T> const& v1, Vector2<T> const& v2) -> Vector2<bool> {
  Vector2<bool> res;
  for (size_t i = 0; i < 2; i++) {
    res.data[i] = v1.data[i] != v2.data[i];
  }
  return res;
}

SE_EXPORT template <class T>
struct Vector3 {
  union {
    T data[3];
    struct {
      T x, y, z;
    };
    struct {
      Vector2<T> xy;
      T _z;
    };
    struct {
      T r, g, b;
    };
    struct {
      T s, t, p;
    };
  };

  Vector3() : x(0), y(0), z(0) {}
  Vector3(T const& _v) : x(_v), y(_v), z(_v) {}
  Vector3(Vector2<T> const& _v) : x(_v.x), y(_v.y), z(0) {}
  Vector3(Vector2<T> const& _v, T const& _z) : x(_v.x), y(_v.y), z(_z) {}
  Vector3(T const& _x, T const& _y, T const& _z = 0) : x(_x), y(_y), z(_z) {}

  auto lengthSquared() const -> float;
  auto length() const -> float;

  template <class U>
  explicit operator Vector3<U>() const;

  explicit operator Vector2<T>() const { return Vector2<T>{x, y}; }

  operator T*() { return data; }
  operator const T* const() { return static_cast<const T*>(data); }
  auto operator[](size_t idx) -> T& { return data[idx]; }
  auto operator[](size_t idx) const -> T const& { return data[idx]; }

  auto at(size_t idx) -> T& { return data[idx]; }
  auto at(size_t idx) const -> T { return data[idx]; }

  auto operator-() const -> Vector3<T>;
  auto operator*(T s) const -> Vector3<T>;
  auto operator/(T s) const -> Vector3<T>;
  auto operator+(Vector3<T> const& v) const -> Vector3<T>;
  auto operator-(Vector3<T> const& v) const -> Vector3<T>;
  auto operator*(Vector3<T> const& v) const -> Vector3<T>;
  auto operator/(Vector3<T> const& v) const -> Vector3<T>;
  auto operator==(Vector3<T> const& v) const -> bool;
  auto operator!=(Vector3<T> const& v) const -> bool;
  auto operator*=(T s) -> Vector3<T>&;
  auto operator/=(T s) -> Vector3<T>&;
  auto operator+=(Vector3<T> const& v) -> Vector3<T>&;
  auto operator-=(Vector3<T> const& v) -> Vector3<T>&;
  auto operator*=(Vector3<T> const& v) -> Vector3<T>&;
};

SE_EXPORT using bvec3 = Vector3<bool>;
SE_EXPORT using vec3 = Vector3<float>;
SE_EXPORT using dvec3 = Vector3<double>;
SE_EXPORT using ivec3 = Vector3<int32_t>;
SE_EXPORT using uvec3 = Vector3<uint32_t>;

template <class T>
template <class U>
Vector3<T>::operator Vector3<U>() const {
  return Vector3<U>(x, y, z);
}

SE_EXPORT template <class S, class T>
auto operator*(S s, Vector3<T> const& v) noexcept -> Vector3<T> {
  return Vector3<T>{T(v.x * s), T(v.y * s), T(v.z * s)};
}

SE_EXPORT template <class T>
inline auto abs(Vector3<T> const& v) noexcept -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; ++i) {
    result.data[i] = std::abs(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto floor(Vector3<T> const& v) noexcept -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; ++i) {
    result.data[i] = std::floor(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto ceil(Vector3<T> const& v) noexcept -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; ++i) {
    result.data[i] = std::ceil(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto dot(Vector3<T> const& x, Vector3<T> const& y) noexcept -> T {
  T result = 0;
  for (size_t i = 0; i < 3; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

SE_EXPORT template <class T>
inline auto absDot(Vector3<T> const& x, Vector3<T> const& y) noexcept -> T {
  return std::abs(dot(x, y));
}

SE_EXPORT template <class T>
inline auto cross(Vector3<T> const& x, Vector3<T> const& y) noexcept
    -> Vector3<T> {
  return Vector3<T>((x[1] * y[2]) - (x[2] * y[1]),
                    (x[2] * y[0]) - (x[0] * y[2]),
                    (x[0] * y[1]) - (x[1] * y[0]));
}

SE_EXPORT template <class T>
inline auto normalize(Vector3<T> const& v) noexcept -> Vector3<T> {
  return v / v.length();
}

SE_EXPORT template <class T>
inline auto sign(Vector3<T> const& v) noexcept -> Vector3<float> {
  return Vector3<float>{sign(v.x), sign(v.y), sign(v.z)};
}

SE_EXPORT template <class T>
inline auto equal(Vector3<T> const& v1, Vector3<T> const& v2) noexcept -> bool {
  return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
}

SE_EXPORT template <class T>
inline auto minComponent(Vector3<T> const& v) noexcept -> T {
  return std::min(std::min(v.x, v.y), v.z);
}

SE_EXPORT template <class T>
inline auto maxComponent(Vector3<T> const& v) noexcept -> T {
  return std::max(std::max(v.x, v.y), v.z);
}

SE_EXPORT template <class T>
inline auto maxDimension(Vector3<T> const& v) noexcept -> size_t {
  return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}
SE_EXPORT template <typename T>
inline auto permute(const Vector3<T>& p, int x, int y, int z) -> Vector3<T> {
  return Vector3<T>(p.data[x], p.data[y], p.data[z]);
}

SE_EXPORT template <class T>
inline auto minDimension(Vector3<T> const& v) noexcept -> size_t {
  return (v.x < v.y) ? ((v.x < v.z) ? 0 : 2) : ((v.y < v.z) ? 1 : 2);
}

SE_EXPORT template <class T>
inline auto max(Vector3<T> const& x, Vector3<T> const& y) noexcept
    -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; ++i) {
    result[i] = std::max(x[i], y[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto min(Vector3<T> const& x, Vector3<T> const& y) noexcept
    -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; ++i) {
    result[i] = std::min(x[i], y[i]);
  }
  return result;
}
SE_EXPORT template <class T>
inline auto permute(Vector3<T> const& v, size_t x, size_t y, size_t z) noexcept
    -> Vector3<T> {
  return Vector3<T>(v[x], v[y], v[z]);
}

SE_EXPORT template <class T>
inline auto distance(Vector3<T> const& p1, Vector3<T> const& p2) noexcept
    -> float {
  return (p1 - p2).length();
}

SE_EXPORT template <class T>
inline auto distanceSquared(Vector3<T> const& p1, Vector3<T> const& p2) noexcept
    -> float {
  return (p1 - p2).lengthSquared();
}

SE_EXPORT template <class T>
inline auto operator*(T s, Vector3<T>& v) -> Vector3<T> {
  return v * s;
}

SE_EXPORT template <class T>
inline auto lerp(float t, Vector3<T> const& x, Vector3<T>& y) noexcept
    -> Vector3<T> {
  return (1 - t) * x + t * y;
}

SE_EXPORT template <class T>
inline auto faceForward(Vector3<T> const& n, Vector3<T> const& v) noexcept
    -> Vector3<T> {
  return (dot(n, v) < 0.0f) ? -n : n;
}

SE_EXPORT template <class T>
inline auto length(Vector3<T> const& x) noexcept -> T {
  return std::sqrt(x.lengthSquared());
}

SE_EXPORT template <class T>
inline auto cos(Vector3<T> const& v) noexcept -> Vector3<T> {
  return Vector3<T>(std::cos(v.x), std::cos(v.y), std::cos(v.z));
}

SE_EXPORT template <class T>
inline auto sin(Vector3<T> const& v) noexcept -> Vector3<T> {
  return Vector3<T>(std::sin(v.x), std::sin(v.y), std::sin(v.z));
}

template <class T>
auto Vector3<T>::lengthSquared() const -> float {
  return x * x + y * y + z * z;
}

template <class T>
auto Vector3<T>::length() const -> float {
  return std::sqrt(lengthSquared());
}

template <class T>
auto Vector3<T>::operator-() const -> Vector3<T> {
  return Vector3<T>{-x, -y, -z};
}

template <class T>
auto Vector3<T>::operator*(T s) const -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; i++) {
    result.data[i] = data[i] * s;
  }
  return result;
}

template <class T>
auto Vector3<T>::operator/(T s) const -> Vector3<T> {
  float inv = 1.f / s;
  Vector3<T> result;
  for (size_t i = 0; i < 3; i++) {
    result.data[i] = data[i] * inv;
  }
  return result;
}

template <class T>
auto Vector3<T>::operator*=(T s) -> Vector3<T>& {
  for (size_t i = 0; i < 3; i++) {
    data[i] *= s;
  }
  return *this;
}

template <class T>
auto Vector3<T>::operator/=(T s) -> Vector3<T>& {
  float inv = 1.f / s;
  for (size_t i = 0; i < 3; i++) {
    data[i] *= inv;
  }
  return *this;
}

template <class T>
auto Vector3<T>::operator+(Vector3<T> const& v) const -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; i++) {
    result.data[i] = data[i] + v.data[i];
  }
  return result;
}

template <class T>
auto Vector3<T>::operator-(Vector3<T> const& v) const -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; i++) {
    result.data[i] = data[i] - v.data[i];
  }
  return result;
}

template <class T>
auto Vector3<T>::operator*(Vector3<T> const& v) const -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; i++) {
    result.data[i] = data[i] * v.data[i];
  }
  return result;
}

template <class T>
auto Vector3<T>::operator/(Vector3<T> const& v) const -> Vector3<T> {
  Vector3<T> result;
  for (size_t i = 0; i < 3; i++) {
    result.data[i] = data[i] / v.data[i];
  }
  return result;
}

template <class T>
auto Vector3<T>::operator+=(Vector3<T> const& v) -> Vector3<T>& {
  for (size_t i = 0; i < 3; i++) {
    data[i] += v.data[i];
  }
  return *this;
}

template <class T>
auto Vector3<T>::operator-=(Vector3<T> const& v) -> Vector3<T>& {
  for (size_t i = 0; i < 3; i++) {
    data[i] -= v.data[i];
  }
  return *this;
}

template <class T>
auto Vector3<T>::operator*=(Vector3<T> const& v) -> Vector3<T>& {
  for (size_t i = 0; i < 3; i++) {
    data[i] *= v.data[i];
  }
  return *this;
}

template <class T>
auto Vector3<T>::operator==(Vector3<T> const& v) const -> bool {
  return (x == v.x) && (y == v.y) && (z == v.z);
}

template <class T>
auto Vector3<T>::operator!=(Vector3<T> const& v) const -> bool {
  return !(*this == v);
}

SE_EXPORT template <class T>
struct Vector4 {
  // __declspec(align(16))
  union {
    T data[4];
    struct {
      T x, y, z, w;
    };
    struct {
      T r, g, b, a;
    };
    struct {
      T s, t, p, q;
    };
  };

  auto lengthSquared() const -> float;
  auto length() const -> float;

  template <class U>
  explicit operator Vector4<U>() const;

  Vector4() : x(0), y(0), z(0), w(0) {}
  Vector4(T const& _v) : x(_v), y(_v), z(_v), w(_v) {}
  Vector4(Vector2<T> const& _v) : x(_v.x), y(_v.y), z(0), w(0) {}
  Vector4(Vector2<T> const& _v, T _z, T _w) : x(_v.x), y(_v.y), z(_z), w(_w) {}
  Vector4(Vector2<T> const& _v1, Vector2<T> const& _v2)
      : x(_v1.x), y(_v1.y), z(_v2.x), w(_v2.y){}
  Vector4(Vector3<T> const& _v) : x(_v.x), y(_v.y), z(_v.z), w(0) {}
  Vector4(Vector3<T> const& _v, T _w) : x(_v.x), y(_v.y), z(_v.z), w(_w) {}
  Vector4(T const& _x, T const& _y, T const& _z = 0, T const& _w = 0)
      : x(_x), y(_y), z(_z), w(_w) {}

  inline auto xyz() noexcept -> Vector3<T> { return Vector3<T>(x, y, z); }
  inline auto xzy() noexcept -> Vector3<T> { return Vector3<T>{x, z, y}; }
  inline auto xyw() noexcept -> Vector3<T> { return Vector3<T>{x, y, w}; }
  inline auto xwy() noexcept -> Vector3<T> { return Vector3<T>{x, w, y}; }
  inline auto xzw() noexcept -> Vector3<T> { return Vector3<T>{x, z, w}; }
  inline auto xwz() noexcept -> Vector3<T> { return Vector3<T>{x, w, z}; }

  operator T*() { return data; }
  operator const T* const() { return static_cast<const T*>(data); }
  auto operator[](size_t idx) -> T& { return data[idx]; }
  auto operator[](size_t idx) const -> T const& { return data[idx]; }

  explicit operator Vector3<T>() { return Vector3<T>{x, y, z}; }

  auto operator-() const -> Vector4<T>;
  auto operator*(T s) const -> Vector4<T>;
  auto operator/(T s) const -> Vector4<T>;
  auto operator*=(T s) -> Vector4<T>&;
  auto operator/=(T s) -> Vector4<T>&;
  auto operator+(Vector4<T> const& v) const -> Vector4<T>;
  auto operator-(Vector4<T> const& v) const -> Vector4<T>;
  auto operator*(Vector4<T> const& v) const -> Vector4<T>;
  auto operator+=(Vector4<T> const& v) -> Vector4<T>&;
  auto operator-=(Vector4<T> const& v) -> Vector4<T>&;
  auto operator*=(Vector4<T> const& v) -> Vector4<T>&;
  auto operator==(Vector4<T> const& v) const -> bool;
  auto operator!=(Vector4<T> const& v) const -> bool;
};

SE_EXPORT using bvec4 = Vector4<bool>;
SE_EXPORT using vec4 = Vector4<float>;
SE_EXPORT using ivec4 = Vector4<int32_t>;
SE_EXPORT using uvec4 = Vector4<uint32_t>;
SE_EXPORT using dvec4 = Vector4<double>;

template <class T>
template <class U>
Vector4<T>::operator Vector4<U>() const {
  return Vector4<U>(x, y, z, w);
}

SE_EXPORT template <class S, class T>
auto operator*(S s, Vector4<T> const& v) noexcept -> Vector4<T> {
  return Vector4<T>{v.x * s, v.y * s, v.z * s, v.w * s};
}

SE_EXPORT template <class T>
inline auto abs(Vector4<T> const& v) noexcept -> T {
  T result;
  for (size_t i = 0; i < 4; ++i) {
    result = std::abs(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto floor(Vector4<T> const& v) noexcept -> Vector4<T> {
  Vector4<T> result;
  for (size_t i = 0; i < 4; ++i) {
    result.data[i] = std::floor(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto ceil(Vector4<T> const& v) noexcept -> Vector4<T> {
  Vector4<T> result;
  for (size_t i = 0; i < 4; ++i) {
    result.data[i] = std::ceil(v.data[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto dot(Vector4<T> const& x, Vector4<T> const& y) noexcept -> T {
  T result;
  for (size_t i = 0; i < 4; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

SE_EXPORT template <class T>
inline auto absDot(Vector4<T> const& x, Vector4<T> const& y) noexcept -> T {
  return std::abs(dot(x, y));
}

SE_EXPORT template <class T>
inline auto normalize(Vector4<T> const& v) noexcept -> Vector4<T> {
  return v / v.length();
}

SE_EXPORT template <class T>
inline auto sign(Vector4<T> const& v) noexcept -> Vector4<float> {
  return Vector4<float>{sign(v.x), sign(v.y), sign(v.z), sign(v.w)};
}

SE_EXPORT template <class T>
inline auto equal(Vector4<T> const& v1, Vector4<T> const& v2) noexcept -> bool {
  return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w;
}

SE_EXPORT template <class T>
inline auto minComponent(Vector4<T> const& v) noexcept -> T {
  return std::min(std::min(v.x, v.y), std::min(v.z, v.w));
}

SE_EXPORT template <class T>
inline auto maxComponent(Vector4<T> const& v) noexcept -> T {
  return std::max(std::max(v.x, v.y), std::max(v.z, v.w));
}

SE_EXPORT template <class T>
inline auto maxDimension(Vector4<T> const& v) noexcept -> size_t {
  return (v.x > v.y)
             ? ((v.x > v.z) ? ((v.x > v.w) ? 0 : 3) : ((v.z > v.w) ? 2 : 3))
             : ((v.y > v.z) ? ((v.y > v.w) ? 1 : 3) : ((v.z > v.w) ? 2 : 3));
}

SE_EXPORT template <class T>
inline auto minDimension(Vector4<T> const& v) noexcept -> size_t {
  return (v.x < v.y)
             ? ((v.x < v.z) ? ((v.x < v.w) ? 0 : 3) : ((v.z < v.w) ? 2 : 3))
             : ((v.y < v.z) ? ((v.y < v.w) ? 1 : 3) : ((v.z < v.w) ? 2 : 3));
}

SE_EXPORT template <class T>
inline auto max(Vector4<T> const& x, Vector4<T> const& y) noexcept
    -> Vector4<T> {
  Vector4<T> result;
  for (size_t i = 0; i < 4; ++i) {
    result[i] = std::max(x[i], y[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto min(Vector4<T> const& x, Vector4<T> const& y) noexcept
    -> Vector4<T> {
  Vector4<T> result;
  for (size_t i = 0; i < 4; ++i) {
    result[i] = std::min(x[i], y[i]);
  }
  return result;
}

SE_EXPORT template <class T>
inline auto permute(Vector4<T> const& v, size_t x, size_t y, size_t z,
                    size_t w) noexcept -> Vector4<T> {
  return Vector4<T>(v[x], v[y], v[z], v[w]);
}

SE_EXPORT template <class T>
inline auto distance(Vector4<T> const& p1, Vector4<T> const& p2) noexcept
    -> float {
  return (p1 - p2).length();
}

SE_EXPORT template <class T>
inline auto distanceSquared(Vector4<T> const& p1, Vector4<T> const& p2) noexcept
    -> float {
  return (p1 - p2).lengthSquared();
}

SE_EXPORT template <class T>
inline auto operator*(T s, Vector4<T>& v) -> Vector4<T> {
  return v * s;
}

SE_EXPORT template <class T>
inline auto lerp(float t, Vector4<T> const& x, Vector4<T>& y) noexcept
    -> Vector4<T> {
  return (1 - t) * x + t * y;
}

SE_EXPORT template <class T>
inline auto length(Vector4<T> const& x) noexcept -> T {
  return std::sqrt(x.lengthSquared());
}

template <class T>
auto Vector4<T>::lengthSquared() const -> float {
  return x * x + y * y + z * z + w * w;
}

template <class T>
auto Vector4<T>::length() const -> float {
  return std::sqrt(lengthSquared());
}

template <class T>
auto Vector4<T>::operator-() const -> Vector4<T> {
  return Vector4<T>{-x, -y, -z, -w};
}

template <class T>
auto Vector4<T>::operator*(T s) const -> Vector4<T> {
  Vector4<T> result;
  for (size_t i = 0; i < 4; i++) {
    result.data[i] = data[i] * s;
  }
  return result;
}

template <class T>
auto Vector4<T>::operator/(T s) const -> Vector4<T> {
  float inv = 1.f / s;
  Vector4<T> result;
  for (size_t i = 0; i < 4; i++) {
    result.data[i] = data[i] * inv;
  }
  return result;
}

template <class T>
auto Vector4<T>::operator*=(T s) -> Vector4<T>& {
  for (size_t i = 0; i < 4; i++) {
    data[i] *= s;
  }
  return *this;
}

template <class T>
auto Vector4<T>::operator/=(T s) -> Vector4<T>& {
  float inv = 1.f / s;
  for (size_t i = 0; i < 4; i++) {
    data[i] *= inv;
  }
  return *this;
}

template <class T>
auto Vector4<T>::operator+(Vector4<T> const& v) const -> Vector4<T> {
  Vector4<T> result;
  for (size_t i = 0; i < 4; i++) {
    result.data[i] = data[i] + v.data[i];
  }
  return result;
}

template <class T>
auto Vector4<T>::operator-(Vector4<T> const& v) const -> Vector4<T> {
  Vector4<T> result;
  for (size_t i = 0; i < 4; i++) {
    result.data[i] = data[i] - v.data[i];
  }
  return result;
}

template <class T>
auto Vector4<T>::operator*(Vector4<T> const& v) const -> Vector4<T> {
  Vector4<T> result;
  for (size_t i = 0; i < 4; i++) {
    result.data[i] = data[i] * v.data[i];
  }
  return result;
}

template <class T>
auto Vector4<T>::operator+=(Vector4<T> const& v) -> Vector4<T>& {
  for (size_t i = 0; i < 4; i++) {
    data[i] += v.data[i];
  }
  return *this;
}

template <class T>
auto Vector4<T>::operator-=(Vector4<T> const& v) -> Vector4<T>& {
  for (size_t i = 0; i < 4; i++) {
    data[i] -= v.data[i];
  }
  return *this;
}

template <class T>
auto Vector4<T>::operator*=(Vector4<T> const& v) -> Vector4<T>& {
  for (size_t i = 0; i < 4; i++) {
    data[i] *= v.data[i];
  }
  return *this;
}

template <class T>
auto Vector4<T>::operator==(Vector4<T> const& v) const -> bool {
  return (x == v.x) && (y == v.y) && (z == v.z) && (w == v.w);
}

template <class T>
auto Vector4<T>::operator!=(Vector4<T> const& v) const -> bool {
  return !(*this == v);
}

SE_EXPORT template<class T>
auto cross(const Vector4<T>& a, const Vector4<T>& b, const Vector4<T>& c)
    -> Vector4<T> {
  // Code adapted from VecLib4d.c in Graphics Gems V
  T d1 = (b[2] * c[3]) - (b[3] * c[2]);
  T d2 = (b[1] * c[3]) - (b[3] * c[1]);
  T d3 = (b[1] * c[2]) - (b[2] * c[1]);
  T d4 = (b[0] * c[3]) - (b[3] * c[0]);
  T d5 = (b[0] * c[2]) - (b[2] * c[0]);
  T d6 = (b[0] * c[1]) - (b[1] * c[0]);
  return Vector4<T>(
      -a[1] * d1 + a[2] * d2 - a[3] * d3, a[0] * d1 - a[2] * d4 + a[3] * d5,
      -a[0] * d2 + a[1] * d4 - a[3] * d6, a[0] * d3 - a[1] * d5 + a[2] * d6);
}

SE_EXPORT template <class T>
struct Matrix2x2 {
  union {
    T data[2][2];
  };
};

SE_EXPORT using mat2 = Matrix2x2<float>;
SE_EXPORT using imat2 = Matrix2x2<int32_t>;
SE_EXPORT using umat2 = Matrix2x2<uint32_t>;

SE_EXPORT template <class T>
struct Matrix3x3 {
  Matrix3x3() = default;
  Matrix3x3(T const mat[3][3]);
  Matrix3x3(Vector3<T> x, Vector3<T> y, Vector3<T> z);
  Matrix3x3(T t00, T t01, T t02, T t10, T t11, T t12, T t20, T t21, T t22);

  auto row(int i) const noexcept -> Vector3<T>;
  auto col(int i) const noexcept -> Vector3<T>;

  union {
    T data[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  };
};

template <class T>
auto Matrix3x3<T>::row(int i) const noexcept -> Vector3<T> {
  if (i < 0 || i >= 3)
    return Vector3<T>{0.};
  else
    return Vector3<T>{data[i][0], data[i][1], data[i][2]};
}

template <class T>
auto Matrix3x3<T>::col(int i) const noexcept->Vector3<T> {
  if (i < 0 || i >= 3)
    return Vector3<T>{0};
  else
    return Vector3<T>{data[0][i], data[1][i], data[2][i]};
}

template <class T>
Matrix3x3<T>::Matrix3x3(T const mat[3][3]) {
  memcpy(&(data[0][0]), &(mat[0][0]), sizeof(T) * 9);
}

template <class T>
Matrix3x3<T>::Matrix3x3(Vector3<T> x, Vector3<T> y, Vector3<T> z) {
  data[0][0] = x.x;
  data[0][1] = x.y;
  data[0][2] = x.z;
  data[1][0] = y.x;
  data[1][1] = y.y;
  data[1][2] = y.z;
  data[2][0] = z.x;
  data[2][1] = z.y;
  data[2][2] = z.z;
}

template <class T>
Matrix3x3<T>::Matrix3x3(T t00, T t01, T t02, T t10, T t11, T t12, T t20, T t21,
                        T t22) {
  data[0][0] = t00;
  data[0][1] = t01;
  data[0][2] = t02;
  data[1][0] = t10;
  data[1][1] = t11;
  data[1][2] = t12;
  data[2][0] = t20;
  data[2][1] = t21;
  data[2][2] = t22;
}

SE_EXPORT template <class T>
inline auto mul(Matrix3x3<T> const& m, Vector3<T> const& v) noexcept
    -> Vector3<T> {
  Vector3<T> s;
  for (size_t i = 0; i < 3; ++i) {
    s.data[i] = m.data[i][0] * v.data[0] + m.data[i][1] * v.data[1] +
                m.data[i][2] * v.data[2];
  }
  return s;
}

SE_EXPORT template <class T>
auto operator*(Matrix3x3<T> const& m, Vector3<T> const& v) -> Vector3<T> {
  return mul<T>(m, v);
}

SE_EXPORT template <class T>
auto operator/(Matrix3x3<T> const& m, T s) -> Matrix3x3<T> {
  return Matrix3x3<T>(m.row(0) / s, m.row(1) / s, m.row(2) / s);
}

SE_EXPORT template <class T>
Matrix3x3<T> adjoint(Matrix3x3<T> const& m) {
  return Matrix3x3<T>(cross(m.row(1), m.row(2)), cross(m.row(2), m.row(0)),
                      cross(m.row(0), m.row(1)));
}

SE_EXPORT template <class T>
Matrix3x3<T> transpose(Matrix3x3<T> const& m) {
  return Matrix3x3<T>(m.col(0), m.col(1), m.col(2));
}

SE_EXPORT template <class T>
double invert(Matrix3x3<T>& inv, Matrix3x3<T> const& m) {
  Matrix3x3<T> A = adjoint(m);
  double d = dot(A.row(0), m.row(0));
  if (d == 0.0) return 0.0;
  inv = transpose(A) / d;
  return d;
}

SE_EXPORT using mat3 = Matrix3x3<float>;
SE_EXPORT using dmat3 = Matrix3x3<double>;
SE_EXPORT using imat3 = Matrix3x3<int32_t>;
SE_EXPORT using umat3 = Matrix3x3<uint32_t>;

SE_EXPORT template <class T>
struct Matrix4x4 {
  Matrix4x4() = default;
  Matrix4x4(T s);
  Matrix4x4(T const mat[4][4]);
  Matrix4x4(Vector4<T> const& a, Vector4<T> const& b, Vector4<T> const& c,
            Vector4<T> const& d);
  Matrix4x4(T t00, T t01, T t02, T t03, T t10, T t11, T t12, T t13, T t20,
            T t21, T t22, T t23, T t30, T t31, T t32, T t33);

  auto operator==(Matrix4x4<T> const& t) const -> bool;
  auto operator!=(Matrix4x4<T> const& t) const -> bool;
  auto operator-() const -> Matrix4x4<T>;
  auto operator+(Matrix4x4<T> const& t) const -> Matrix4x4<T>;
  auto operator-(Matrix4x4<T> const& t) const -> Matrix4x4<T>;

  operator Matrix3x3<T>() const;

  T data[4][4] = {
      {1, 0, 0, 0},
      {0, 1, 0, 0},
      {0, 0, 1, 0},
      {0, 0, 0, 1},
  };

  auto row(int i) const noexcept -> Vector4<T>;
  auto col(int i) const noexcept -> Vector4<T>;
  auto set_row(int i, Vector4<T> const& x) noexcept -> void;

  static inline auto translate(Vector3<T> const& delta) noexcept
      -> Matrix4x4<T>;
  static inline auto scale(float x, float y, float z) noexcept -> Matrix4x4<T>;
  static inline auto scale(Vector3<T> const& scale) noexcept -> Matrix4x4<T>;
  static inline auto rotateX(float theta) noexcept -> Matrix4x4<T>;
  static inline auto rotateY(float theta) noexcept -> Matrix4x4<T>;
  static inline auto rotateZ(float theta) noexcept -> Matrix4x4<T>;
  static inline auto rotate(float theta, vec3 const& axis) noexcept
      -> Matrix4x4<T>;
};

template <class T>
auto Matrix4x4<T>::row(int i) const noexcept -> Vector4<T> {
  if (i < 0 || i >= 4)
    return Vector4<T>{0};
  else
    return Vector4<T>{data[i][0], data[i][1], data[i][2], data[i][3]};
}

template <class T>
auto Matrix4x4<T>::col(int i) const noexcept->Vector4<T> {
  if (i < 0 || i >= 4)
    return Vector4<T>{nanf};
  else
    return Vector4<T>{data[0][i], data[1][i], data[2][i], data[3][i]};
}

template <class T>
auto Matrix4x4<T>::set_row(int i, Vector4<T> const& x) noexcept -> void {
  if (i < 0 || i >= 4)
    return;
  else {
    data[i][0] = x.x;
    data[i][1] = x.y;
    data[i][2] = x.z;
    data[i][3] = x.w;
  }
}

template <class T>
Matrix4x4<T>::Matrix4x4(T s) {
  data[0][0] = s;
  data[0][1] = 0;
  data[0][2] = 0;
  data[0][3] = 0;
  data[1][0] = 0;
  data[1][1] = s;
  data[1][2] = 0;
  data[1][3] = 0;
  data[2][0] = 0;
  data[2][1] = 0;
  data[2][2] = s;
  data[2][3] = 0;
  data[3][0] = 0;
  data[3][1] = 0;
  data[3][2] = 0;
  data[3][3] = 1;
}

template <class T>
Matrix4x4<T>::Matrix4x4(T const mat[4][4]) {
  memcpy(&(data[0][0]), &(mat[0][0]), sizeof(T) * 16);
}

template <class T>
Matrix4x4<T>::Matrix4x4(Vector4<T> const& a, Vector4<T> const& b,
    Vector4<T> const& c, Vector4<T> const& d) {
  data[0][0] = a.x;
  data[0][1] = a.y;
  data[0][2] = a.z;
  data[0][3] = a.w;
  data[1][0] = b.x;
  data[1][1] = b.y;
  data[1][2] = b.z;
  data[1][3] = b.w;
  data[2][0] = c.x;
  data[2][1] = c.y;
  data[2][2] = c.z;
  data[2][3] = c.w;
  data[3][0] = d.x;
  data[3][1] = d.y;
  data[3][2] = d.z;
  data[3][3] = d.w;
}

template <class T>
Matrix4x4<T>::Matrix4x4(T t00, T t01, T t02, T t03, T t10, T t11, T t12, T t13,
                        T t20, T t21, T t22, T t23, T t30, T t31, T t32,
                        T t33) {
  data[0][0] = t00;
  data[0][1] = t01;
  data[0][2] = t02;
  data[0][3] = t03;
  data[1][0] = t10;
  data[1][1] = t11;
  data[1][2] = t12;
  data[1][3] = t13;
  data[2][0] = t20;
  data[2][1] = t21;
  data[2][2] = t22;
  data[2][3] = t23;
  data[3][0] = t30;
  data[3][1] = t31;
  data[3][2] = t32;
  data[3][3] = t33;
}

template <class T>
auto Matrix4x4<T>::operator==(Matrix4x4<T> const& t) const -> bool {
  return (memcmp(&(data[0][0]), &(t.data[0][0]), sizeof(T) * 16) == 0) ? true
                                                                       : false;
}

template <class T>
auto Matrix4x4<T>::operator!=(Matrix4x4<T> const& t) const -> bool {
  return !(*this == t);
}

template <class T>
auto Matrix4x4<T>::operator-() const -> Matrix4x4<T> {
  return Matrix4x4<T>{-data[0][0], -data[0][1], -data[0][2], -data[0][3],
                      -data[1][0], -data[1][1], -data[1][2], -data[1][3],
                      -data[2][0], -data[2][1], -data[2][2], -data[2][3],
                      -data[3][0], -data[3][1], -data[3][2], -data[3][3]};
}

template <class T>
auto Matrix4x4<T>::operator+(Matrix4x4<T> const& t) const -> Matrix4x4<T> {
  return Matrix4x4<T>{data[0][0]+t.data[0][0], data[0][1]+t.data[0][1], data[0][2]+t.data[0][2], data[0][3]+t.data[0][3],
                      data[1][0]+t.data[1][0], data[1][1]+t.data[1][1], data[1][2]+t.data[1][2], data[1][3]+t.data[1][3],
                      data[2][0]+t.data[2][0], data[2][1]+t.data[2][1], data[2][2]+t.data[2][2], data[2][3]+t.data[2][3],
                      data[3][0]+t.data[3][0], data[3][1]+t.data[3][1], data[3][2]+t.data[3][2], data[3][3]+t.data[3][3]};
}

template <class T>
auto Matrix4x4<T>::operator-(Matrix4x4<T> const& t) const -> Matrix4x4<T> {
  return Matrix4x4<T>{data[0][0]-t.data[0][0], data[0][1]-t.data[0][1], data[0][2]-t.data[0][2], data[0][3]-t.data[0][3],
                      data[1][0]-t.data[1][0], data[1][1]-t.data[1][1], data[1][2]-t.data[1][2], data[1][3]-t.data[1][3],
                      data[2][0]-t.data[2][0], data[2][1]-t.data[2][1], data[2][2]-t.data[2][2], data[2][3]-t.data[2][3],
                      data[3][0]-t.data[3][0], data[3][1]-t.data[3][1], data[3][2]-t.data[3][2], data[3][3]-t.data[3][3]};
}

template <class T>
Matrix4x4<T>::operator Matrix3x3<T>() const {
  return Matrix3x3<T>{
      data[0][0], data[0][1], data[0][2], data[1][0], data[1][1],
      data[1][2], data[2][0], data[2][1], data[2][2],
  };
}

SE_EXPORT template <class T>
inline auto transpose(Matrix4x4<T> const& m) noexcept -> Matrix4x4<T> {
  return Matrix4x4<T>(m.data[0][0], m.data[1][0], m.data[2][0], m.data[3][0],
                      m.data[0][1], m.data[1][1], m.data[2][1], m.data[3][1],
                      m.data[0][2], m.data[1][2], m.data[2][2], m.data[3][2],
                      m.data[0][3], m.data[1][3], m.data[2][3], m.data[3][3]);
}

SE_EXPORT template <class T>
inline auto trace(Matrix4x4<T> const& m) noexcept -> T {
  return m.data[0][0] + m.data[1][1] + m.data[2][2] + m.data[3][3];
}

SE_EXPORT template <class T>
inline auto mul(Matrix4x4<T> const& m1, Matrix4x4<T> const& m2) noexcept
    -> Matrix4x4<T> {
  Matrix4x4<T> s;
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 4; ++j)
      s.data[i][j] =
          m1.data[i][0] * m2.data[0][j] + m1.data[i][1] * m2.data[1][j] +
          m1.data[i][2] * m2.data[2][j] + m1.data[i][3] * m2.data[3][j];
  return s;
}

SE_EXPORT template <class T>
inline auto operator*(Matrix4x4<T> const& m1, Matrix4x4<T> const& m2) -> Matrix4x4<T> {
  return mul<T>(m1, m2);
}

SE_EXPORT template <>
inline auto mul(Matrix4x4<float> const& m1, Matrix4x4<float> const& m2) noexcept
    -> Matrix4x4<float> {
  Matrix4x4<float> s;

  __m128 v, result;
  __m128 const mrow0 = _mm_load_ps(&m2.data[0][0]);
  __m128 const mrow1 = _mm_load_ps(&m2.data[1][0]);
  __m128 const mrow2 = _mm_load_ps(&m2.data[2][0]);
  __m128 const mrow3 = _mm_load_ps(&m2.data[3][0]);

  for (int i = 0; i < 4; ++i) {
    v = _mm_load_ps(&m1.data[i][0]);
    result = _mm_mul_ps(_mm_replicate_x_ps(v), mrow0);
    result = _mm_add_ps(_mm_mul_ps(_mm_replicate_y_ps(v), mrow1), result);
    result = _mm_add_ps(_mm_mul_ps(_mm_replicate_z_ps(v), mrow2), result);
    result = _mm_add_ps(_mm_mul_ps(_mm_replicate_w_ps(v), mrow3), result);
    _mm_store_ps(&s.data[i][0], result);
  }
  return s;
}

SE_EXPORT template <>
inline auto operator*(Matrix4x4<float> const& m1, Matrix4x4<float> const& m2)
    -> Matrix4x4<float> {
  return mul<float>(m1, m2);
}

SE_EXPORT template <class T>
inline auto mul(Matrix4x4<T> const& m, Vector4<T> const& v) noexcept
    -> Vector4<T> {
  Vector4<T> s;
  for (size_t i = 0; i < 4; ++i) {
    s.data[i] = m.data[i][0] * v.data[0] + m.data[i][1] * v.data[1] +
                m.data[i][2] * v.data[2] + m.data[i][3] * v.data[3];
  }
  return s;
}

SE_EXPORT template <class T>
inline auto operator*(Matrix4x4<T> const& m, Vector4<T> const& v)->Vector4<T> {
  return mul<T>(m, v);
}

SE_EXPORT template <>
inline auto mul(Matrix4x4<float> const& m, Vector4<float> const& v) noexcept
    -> Vector4<float> {
  Vector4<float> s;
  __m128 mrow0, mrow1, mrow2, mrow3;
  __m128 acc_0, acc_1, acc_2, acc_3;
  __m128 const vcol = _mm_load_ps(&v.data[0]);

  mrow0 = _mm_load_ps(&(m.data[0][0]));
  mrow1 = _mm_load_ps(&(m.data[1][0]));
  mrow2 = _mm_load_ps(&(m.data[2][0]));
  mrow3 = _mm_load_ps(&(m.data[3][0]));

  acc_0 = _mm_mul_ps(mrow0, vcol);
  acc_1 = _mm_mul_ps(mrow1, vcol);
  acc_2 = _mm_mul_ps(mrow2, vcol);
  acc_3 = _mm_mul_ps(mrow3, vcol);

  acc_0 = _mm_hadd_ps(acc_0, acc_1);
  acc_2 = _mm_hadd_ps(acc_2, acc_3);
  acc_0 = _mm_hadd_ps(acc_0, acc_2);
  _mm_store_ps(&s.data[0], acc_0);
  return s;
}

SE_EXPORT template <>
inline auto operator*(Matrix4x4<float> const& m, Vector4<float> const& v)
    -> Vector4<float> {
  return mul<float>(m, v);
}

SE_EXPORT template <class T>
inline auto determinant(Matrix4x4<T> const& m) noexcept -> double {
  double Result[4][4];
  double tmp[12]; /* temp array for pairs */
  double src[16]; /* array of transpose source matrix */
  double det;     /* determinant */
  /* transpose matrix */
  for (int i = 0; i < 4; i++) {
    src[i + 0] = m.data[i][0];
    src[i + 4] = m.data[i][1];
    src[i + 8] = m.data[i][2];
    src[i + 12] = m.data[i][3];
  }
  /* calculate pairs for first 8 elements (cofactors) */
  tmp[0] = src[10] * src[15];
  tmp[1] = src[11] * src[14];
  tmp[2] = src[9] * src[15];
  tmp[3] = src[11] * src[13];
  tmp[4] = src[9] * src[14];
  tmp[5] = src[10] * src[13];
  tmp[6] = src[8] * src[15];
  tmp[7] = src[11] * src[12];
  tmp[8] = src[8] * src[14];
  tmp[9] = src[10] * src[12];
  tmp[10] = src[8] * src[13];
  tmp[11] = src[9] * src[12];
  /* calculate first 8 elements (cofactors) */
  Result[0][0] = tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7];
  Result[0][0] -= tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7];
  Result[0][1] = tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7];
  Result[0][1] -= tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7];
  Result[0][2] = tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7];
  Result[0][2] -= tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7];
  Result[0][3] = tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6];
  Result[0][3] -= tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6];
  Result[1][0] = tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3];
  Result[1][0] -= tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3];
  Result[1][1] = tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3];
  Result[1][1] -= tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3];
  Result[1][2] = tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3];
  Result[1][2] -= tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3];
  Result[1][3] = tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2];
  Result[1][3] -= tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2];
  /* calculate pairs for second 8 elements (cofactors) */
  tmp[0] = src[2] * src[7];
  tmp[1] = src[3] * src[6];
  tmp[2] = src[1] * src[7];
  tmp[3] = src[3] * src[5];
  tmp[4] = src[1] * src[6];
  tmp[5] = src[2] * src[5];

  tmp[6] = src[0] * src[7];
  tmp[7] = src[3] * src[4];
  tmp[8] = src[0] * src[6];
  tmp[9] = src[2] * src[4];
  tmp[10] = src[0] * src[5];
  tmp[11] = src[1] * src[4];
  /* calculate second 8 elements (cofactors) */
  Result[2][0] = tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15];
  Result[2][0] -= tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15];
  Result[2][1] = tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15];
  Result[2][1] -= tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15];
  Result[2][2] = tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15];
  Result[2][2] -= tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15];
  Result[2][3] = tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14];
  Result[2][3] -= tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14];
  Result[3][0] = tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9];
  Result[3][0] -= tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10];
  Result[3][1] = tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10];
  Result[3][1] -= tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8];
  Result[3][2] = tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8];
  Result[3][2] -= tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9];
  Result[3][3] = tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9];
  Result[3][3] -= tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8];
  /* calculate determinant */
  det = src[0] * Result[0][0] + src[1] * Result[0][1] + src[2] * Result[0][2] +
        src[3] * Result[0][3];
  return det;
}

SE_EXPORT template <class T>
inline auto adjoint(Matrix4x4<T> const& m) noexcept -> Matrix4x4<T> {
  Matrix4x4<T> A;
  A.set_row(0, cross(m.row(1), m.row(2), m.row(3)));
  A.set_row(1, cross(-m.row(0), m.row(2), m.row(3)));
  A.set_row(2, cross(m.row(0), m.row(1), m.row(3)));
  A.set_row(3, cross(-m.row(0), m.row(1), m.row(2)));
  return A;
}

SE_EXPORT template <class T>
inline auto inverse(Matrix4x4<T> const& m) noexcept -> Matrix4x4<T> {
  //  Inversion by Cramer's rule.  Code taken from an Intel publication
  double Result[4][4];
  double tmp[12]; /* temp array for pairs */
  double src[16]; /* array of transpose source matrix */
  double det;     /* determinant */
  /* transpose matrix */
  for (int i = 0; i < 4; i++) {
    src[i + 0] = m.data[i][0];
    src[i + 4] = m.data[i][1];
    src[i + 8] = m.data[i][2];
    src[i + 12] = m.data[i][3];
  }
  /* calculate pairs for first 8 elements (cofactors) */
  tmp[0] = src[10] * src[15];
  tmp[1] = src[11] * src[14];
  tmp[2] = src[9] * src[15];
  tmp[3] = src[11] * src[13];
  tmp[4] = src[9] * src[14];
  tmp[5] = src[10] * src[13];
  tmp[6] = src[8] * src[15];
  tmp[7] = src[11] * src[12];
  tmp[8] = src[8] * src[14];
  tmp[9] = src[10] * src[12];
  tmp[10] = src[8] * src[13];
  tmp[11] = src[9] * src[12];
  /* calculate first 8 elements (cofactors) */
  Result[0][0] = tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7];
  Result[0][0] -= tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7];
  Result[0][1] = tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7];
  Result[0][1] -= tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7];
  Result[0][2] = tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7];
  Result[0][2] -= tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7];
  Result[0][3] = tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6];
  Result[0][3] -= tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6];
  Result[1][0] = tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3];
  Result[1][0] -= tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3];
  Result[1][1] = tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3];
  Result[1][1] -= tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3];
  Result[1][2] = tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3];
  Result[1][2] -= tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3];
  Result[1][3] = tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2];
  Result[1][3] -= tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2];
  /* calculate pairs for second 8 elements (cofactors) */
  tmp[0] = src[2] * src[7];
  tmp[1] = src[3] * src[6];
  tmp[2] = src[1] * src[7];
  tmp[3] = src[3] * src[5];
  tmp[4] = src[1] * src[6];
  tmp[5] = src[2] * src[5];

  tmp[6] = src[0] * src[7];
  tmp[7] = src[3] * src[4];
  tmp[8] = src[0] * src[6];
  tmp[9] = src[2] * src[4];
  tmp[10] = src[0] * src[5];
  tmp[11] = src[1] * src[4];
  /* calculate second 8 elements (cofactors) */
  Result[2][0] = tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15];
  Result[2][0] -= tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15];
  Result[2][1] = tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15];
  Result[2][1] -= tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15];
  Result[2][2] = tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15];
  Result[2][2] -= tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15];
  Result[2][3] = tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14];
  Result[2][3] -= tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14];
  Result[3][0] = tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9];
  Result[3][0] -= tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10];
  Result[3][1] = tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10];
  Result[3][1] -= tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8];
  Result[3][2] = tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8];
  Result[3][2] -= tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9];
  Result[3][3] = tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9];
  Result[3][3] -= tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8];
  /* calculate determinant */
  det = src[0] * Result[0][0] + src[1] * Result[0][1] + src[2] * Result[0][2] +
        src[3] * Result[0][3];
  /* calculate matrix inverse */
  det = 1.0f / det;

  Matrix4x4<T> result;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result.data[i][j] = float(Result[i][j] * det);
    }
  }
  return result;
}

template <class T>
inline auto Matrix4x4<T>::translate(Vector3<T> const& delta) noexcept
    -> Matrix4x4<T> {
  return Matrix4x4<T>(1, 0, 0, delta.x, 0, 1, 0, delta.y, 0, 0, 1, delta.z, 0,
                      0, 0, 1);
}

template <class T>
inline auto Matrix4x4<T>::scale(float x, float y, float z) noexcept
    -> Matrix4x4<T> {
  return Matrix4x4<T>(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
}

template <class T>
inline auto Matrix4x4<T>::scale(Vector3<T> const& scale) noexcept
    -> Matrix4x4<T> {
  return Matrix4x4<T>(scale.x, 0, 0, 0, 0, scale.y, 0, 0, 0, 0, scale.z, 0, 0,
                      0, 0, 1);
}

template <class T>
inline auto Matrix4x4<T>::rotateX(float theta) noexcept -> Matrix4x4<T> {
  float sinTheta = std::sin(Math::radians(theta));
  float cosTheta = std::cos(Math::radians(theta));
  return Matrix4x4<T>(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta,
                      cosTheta, 0, 0, 0, 0, 1);
}

template <class T>
inline auto Matrix4x4<T>::rotateY(float theta) noexcept -> Matrix4x4<T> {
  float sinTheta = std::sin(Math::radians(theta));
  float cosTheta = std::cos(Math::radians(theta));
  Matrix4x4<T> m(T(cosTheta), 0, T(sinTheta), 0, 0, 1, 0, 0, T(-sinTheta), 0,
                 T(cosTheta), 0, 0, 0, 0, 1);
  return m;
}

template <class T>
inline auto Matrix4x4<T>::rotateZ(float theta) noexcept -> Matrix4x4<T> {
  float sinTheta = std::sin(Math::radians(theta));
  float cosTheta = std::cos(Math::radians(theta));
  Matrix4x4<T> m(cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 1);
  return m;
}

template <class T>
inline auto Matrix4x4<T>::rotate(float theta, vec3 const& axis) noexcept
    -> Matrix4x4<T> {
  vec3 a = normalize(axis);
  float sinTheta = std::sin(Math::radians(theta));
  float cosTheta = std::cos(Math::radians(theta));
  Matrix4x4<T> m;
  // Compute rotation of first basis vector
  m.data[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
  m.data[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
  m.data[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
  m.data[0][3] = 0;
  // Compute rotations of second and third basis vectors
  m.data[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
  m.data[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
  m.data[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
  m.data[1][3] = 0;

  m.data[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
  m.data[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
  m.data[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
  m.data[2][3] = 0;
  return m;
}

SE_EXPORT using mat4 = Matrix4x4<float>;
SE_EXPORT using dmat4 = Matrix4x4<double>;
SE_EXPORT using imat4 = Matrix4x4<int32_t>;
SE_EXPORT using umat4 = Matrix4x4<uint32_t>;


SE_EXPORT template <class T>
	struct Point2 :public Vector2<T>
{
	Point2(Vector2<T> const& v = { 0,0 })
		:Vector2<T>(v) {}

	Point2(T const& x, T const& y)
		:Vector2<T>(x, y) {}

	template<class U>
	Point2(Vector2<U> const& v = { 0,0,0 })
		: Vector2<T>((T)v.x, (T)v.y) {}

	template <typename U>
	explicit Point2(Point2<U> const& p)
		:Vector2<T>((T)p.x, (T)p.y) {}

	template <typename U>
	explicit operator Point2<U>() const { return Point2<U>((U)this->x, (U)this->y); }

	auto operator+(Vector2<T> const& a) const -> Point2 { return Point2{ this->x + a.x,this->y + a.y }; }
	auto operator-(Vector2<T> const& a) const -> Point2 { return Point2{ this->x - a.x,this->y - a.y }; }
};

SE_EXPORT template <class T>
inline auto max(Point2<T> const& a, Point2<T> const& b) noexcept -> Point2<T> {
	return Point2<T>{std::max(a.x, b.x), std::max(a.y, b.y)};
}

SE_EXPORT template <class T>
inline auto min(Point2<T> const& a, Point2<T> const& b) noexcept -> Point2<T> {
	return Point2<T>{std::min(a.x, b.x), std::min(a.y, b.y)};
}

SE_EXPORT using point2 = Point2<float>;
SE_EXPORT using ipoint2 = Point2<int32_t>;
SE_EXPORT using upoint2 = Point2<uint32_t>;
    
SE_EXPORT template <class T>
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

SE_EXPORT using point3 = Point3<float>;
SE_EXPORT using ipoint3 = Point3<int32_t>;
SE_EXPORT using upoint3 = Point3<uint32_t>;
    
SE_EXPORT template <class T>
	struct Normal3 :public Vector3<T>
{
	Normal3(T const& _x, T const& _y, T const& _z = 0)
		:Vector3<T>(_x, _y, _z) {}

	Normal3(Vector3<T> const& v = { 0,0,0 })
		:Vector3<T>(v) {}

	template <typename U>
	explicit Normal3(Normal3<U> const& p)
		:Vector3<T>((T)p.x, (T)p.y, (T)p.z) {}

	auto operator*=(T s) -> Normal3<T>&;
	auto operator-() const -> Normal3<T>;

	template <typename U>
	explicit operator Vector3<U>() const { return Vector3<U>(this->x, this->y, this->z); }
};

SE_EXPORT using normal3 = Normal3<float>;
SE_EXPORT using inormal3 = Normal3<int32_t>;
SE_EXPORT using unormal3 = Normal3<uint32_t>;

SE_EXPORT template <class T>
inline auto faceforward(Normal3<T> const& n, Vector3<T> const& v) {
	return (Math::dot(n, v) < 0.f) ? -n : n;
}

template <class T>
auto Normal3<T>::operator*=(T s) -> Normal3<T>& {
	this->x *= s; this->y *= s; this->z *= s;
	return *this;
}

template <class T>
auto Normal3<T>::operator-() const->Normal3<T> {
	return Normal3<T>{-this->x, -this->y, -this->z};
}

SE_EXPORT struct ray3 {
    ray3() : tMax(float_infinity) {}
    ray3(Math::point3 const& o, Math::vec3 const& d,
            float tMax = float_infinity)
        : o(o), d(d), tMax(tMax) {}

    inline auto operator()(float t) const -> Math::point3;

    /** origin */
    point3 o;
    /** direction */
    vec3 d;
    /** restrict the ray to segment [0,r(tMax)]*/
    mutable float tMax;
};

inline auto ray3::operator()(float t) const->Math::point3 {
 return o + d * t;
}
}  // namespace SIByL::Math


namespace SIByL::Math {
SE_EXPORT template <class T>
struct Bounds2 {
    Bounds2();
    Bounds2(Point2<T> const& p);
    Bounds2(Point2<T> const& p1, Point2<T> const& p2);

    auto operator[](uint32_t i) const -> Point2<T> const&;
    auto operator[](uint32_t i) -> Point2<T>&;

    template <typename U>
    explicit operator Bounds2<U>() const {
    return Bounds2<U>((Point2<U>)this->pMin, (Point2<U>)this->pMax);
    }

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
SE_EXPORT using bounds2 = Bounds2<float>;
SE_EXPORT using ibounds2 = Bounds2<int32_t>;

// iterator for ibounds2
// -----------------------
class ibounds2Iterator : public std::forward_iterator_tag {
   public:
    ibounds2Iterator(const ibounds2& b, const ipoint2& pt)
        : p(pt), bounds(&b) {}

    auto operator++() -> ibounds2Iterator;
    auto operator++(int) -> ibounds2Iterator;
    auto operator==(const ibounds2Iterator& bi) const -> bool;
    auto operator!=(const ibounds2Iterator& bi) const -> bool;
    auto operator*() const noexcept -> ipoint2 { return p; }

   private:
    auto advance() noexcept -> void;
    ipoint2 p;
    ibounds2 const* bounds;
};
SE_EXPORT inline auto begin(ibounds2 const& b) -> ibounds2Iterator;
SE_EXPORT inline auto end(ibounds2 const& b) -> ibounds2Iterator;

// template impl
// -----------------------
template <class T>
Bounds2<T>::Bounds2() {
    T minNum = std::numeric_limits<T>::lowest();
    T maxNum = std::numeric_limits<T>::max();

    pMin = Vector2<T>(maxNum);
    pMax = Vector2<T>(minNum);
}

template <class T>
Bounds2<T>::Bounds2(Point2<T> const& p) {
    pMin = p;
    pMax = p;
}

template <class T>
Bounds2<T>::Bounds2(Point2<T> const& p1, Point2<T> const& p2) {
    pMin = Vector2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
    pMax = Vector2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
}

template <class T>
auto Bounds2<T>::operator[](uint32_t i) const -> Point2<T> const& {
    return (i == 0) ? pMin : pMax;
}

template <class T>
auto Bounds2<T>::operator[](uint32_t i) -> Point2<T>& {
    return (i == 0) ? pMin : pMax;
}

template <class T>
auto Bounds2<T>::corner(uint32_t c) const -> Point2<T> {
    return Point2<T>((*this)[(c & 1)].x, (*this)[(c & 2) ? 1 : 0].y);
}

template <class T>
auto Bounds2<T>::diagonal() const -> Vector2<T> {
    return pMax - pMin;
}

template <class T>
auto Bounds2<T>::surfaceArea() const -> T {
    Vector2<T> d = diagonal();
    return d.x * d.y;
}

template <class T>
auto Bounds2<T>::maximumExtent() const -> uint32_t {
    Vector3<T> d = diagonal();
    if (d.x > d.y)
    return 0;
    else
    return 1;
}

template <class T>
auto Bounds2<T>::lerp(Point2<T> const& t) const -> Point2<T> {
    return Point2<T>(
        {std::lerp(pMin.x, pMax.x, t.x), std::lerp(pMin.y, pMax.y, t.y)});
}

template <class T>
auto Bounds2<T>::offset(Point2<T> const& p) const -> Vector2<T> {
    Vector2<T> o = p - pMin;
    if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
    if (pMax.y > pMin.y) o.x /= pMax.y - pMin.y;
    return o;
}

template <class T>
auto Bounds2<T>::boundingCircle(Point2<T>* center, float* radius) const
    -> void {
    *center = (pMin + pMax) / 2;
    *radius = inside(*center, *this) ? distance(*center, pMax) : 0;
}

SE_EXPORT template <class T>
inline auto unionPoint(Bounds2<T> const& b, Point2<T> const& p) noexcept
    -> Bounds2<T> {
    return Bounds2<T>(
        Point2<T>(std::min(b.pMin.x, p.x), std::min(b.pMin.y, p.y)),
        Point2<T>(std::max(b.pMax.x, p.x), std::max(b.pMax.y, p.y)));
}

SE_EXPORT template <class T>
inline auto unionBounds(Bounds2<T> const& b1, Bounds2<T> const& b2) noexcept
    -> Bounds2<T> {
    return Bounds2<T>(Point2<T>(std::min(b1.pMin.x, b2.pMin.x),
                                std::min(b1.pMin.y, b2.pMin.y)),
                      Point2<T>(std::max(b1.pMax.x, b2.pMax.x),
                                std::max(b1.pMax.y, b2.pMax.y)));
}

SE_EXPORT template <class T>
inline auto intersect(Bounds2<T> const& b1, Bounds2<T> const& b2) noexcept
    -> Bounds2<T> {
    return Bounds2<T>(Point2<T>(std::max(b1.pMin.x, b2.pMin.x),
                                std::max(b1.pMin.y, b2.pMin.y)),
                      Point2<T>(std::min(b1.pMax.x, b2.pMax.x),
                                std::min(b1.pMax.y, b2.pMax.y)));
}

SE_EXPORT template <class T>
inline auto overlaps(Bounds2<T> const& b1, Bounds2<T> const& b2) noexcept
    -> Bounds2<T> {
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    return (x && y);
}

SE_EXPORT template <class T>
inline auto inside(Point2<T> const& p, Bounds2<T> const& b) noexcept -> bool {
    return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
            p.y <= b.pMax.y);
}

SE_EXPORT template <class T>
inline auto insideExclusive(Point2<T> const& p, Bounds2<T> const& b) noexcept
    -> bool {
    return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y &&
            p.y < b.pMax.y);
}

SE_EXPORT template <class T>
inline auto expand(Bounds2<T> const& b, float delta) noexcept -> bool {
    return Bounds2<T>(b.pMin - Vector2<T>(delta), b.pMax + Vector2<T>(delta));
}

inline auto begin(ibounds2 const& b) -> ibounds2Iterator {
    return ibounds2Iterator(b, b.pMin);
}

inline auto end(ibounds2 const& b) -> ibounds2Iterator {
    // Normally, the ending point is at the minimum x value and one past
    // the last valid y value.
    ipoint2 pEnd = ivec2(b.pMin.x, b.pMax.y);
    // However, if the bounds are degenerate, override the end point to
    // equal the start point so that any attempt to iterate over the bounds
    // exits out immediately.
    if (b.pMin.x >= b.pMax.x || b.pMin.y >= b.pMax.y) pEnd = b.pMin;
    return ibounds2Iterator(b, pEnd);
}
}  // namespace SIByL::Math

namespace SIByL::Math {
SE_EXPORT template <class T>
struct Bounds3 {
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
    auto lerp(Point3<T> const& t) const -> Point3<T>;
    auto offset(Point3<T> const& p) const -> Vector3<T>;
    auto boundingSphere(Point3<T>* center, float* radius) const -> void;

    /** Test a ray-AABB intersection */
    inline auto intersectP(ray3 const& ray, float* hitt0, float* hitt1) const
        -> bool;
    /** Test a ray-AABB intersection, using precomputed values indicating
     * negativity of each components. */
    inline auto intersectP(ray3 const& ray, vec3 const& invDir,
                           ivec3 const& dirIsNeg) const -> bool;

    Point3<T> pMin;
    Point3<T> pMax;
};

template <class T>
Bounds3<T>::Bounds3() {
    T minNum = std::numeric_limits<T>::lowest();
    T maxNum = std::numeric_limits<T>::max();

    pMin = Vector3<T>(maxNum);
    pMax = Vector3<T>(minNum);
}

template <class T>
Bounds3<T>::Bounds3(Point3<T> const& p) {
    pMin = p;
    pMax = p;
}

template <class T>
Bounds3<T>::Bounds3(Point3<T> const& p1, Point3<T> const& p2) {
    pMin = Vector3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                      std::min(p1.z, p2.z));
    pMax = Vector3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
                      std::max(p1.z, p2.z));
}

template <class T>
auto Bounds3<T>::operator[](uint32_t i) const -> Point3<T> const& {
    return (i == 0) ? pMin : pMax;
}

template <class T>
auto Bounds3<T>::operator[](uint32_t i) -> Point3<T>& {
    return (i == 0) ? pMin : pMax;
}

template <class T>
auto Bounds3<T>::corner(uint32_t c) const -> Point3<T> {
    return Point3<T>((*this)[(c & 1)].x, (*this)[(c & 2) ? 1 : 0].y,
                     (*this)[(c & 4) ? 1 : 0].z);
}

template <class T>
auto Bounds3<T>::diagonal() const -> Vector3<T> {
    return pMax - pMin;
}

template <class T>
auto Bounds3<T>::surfaceArea() const -> T {
    Vector3<T> d = diagonal();
    return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
}

template <class T>
auto Bounds3<T>::volume() const -> T {
    Vector3<T> d = diagonal();
    return d.x * d.y * d.z;
}

template <class T>
auto Bounds3<T>::maximumExtent() const -> uint32_t {
    Vector3<T> d = diagonal();
    if (d.x > d.y && d.x > d.z)
    return 0;
    else if (d.y > d.z)
    return 1;
    else
    return 2;
}

template <class T>
auto Bounds3<T>::lerp(Point3<T> const& t) const -> Point3<T> {
    return Point3<T>({std::lerp(pMin.x, pMax.x, t.x),
                      std::lerp(pMin.y, pMax.y, t.y),
                      std::lerp(pMin.z, pMax.z, t.z)});
}

template <class T>
auto Bounds3<T>::offset(Point3<T> const& p) const -> Vector3<T> {
    Vector3<T> o = p - pMin;
    if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
    if (pMax.y > pMin.y) o.x /= pMax.y - pMin.y;
    if (pMax.z > pMin.z) o.x /= pMax.z - pMin.z;
    return o;
}

template <class T>
auto Bounds3<T>::boundingSphere(Point3<T>* center, float* radius) const
    -> void {
    *center = (pMin + pMax) / 2;
    *radius = inside(*center, *this) ? distance(*center, pMax) : 0;
}

template <class T>
auto Bounds3<T>::intersectP(ray3 const& ray, float* hitt0, float* hitt1) const
    -> bool {
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
auto Bounds3<T>::intersectP(ray3 const& ray, vec3 const& invDir,
                            ivec3 const& dirIsNeg) const -> bool {
    Bounds3<float> const& bounds = *this;
    // Check for ray intersection against x and y slabs
    float tMin = (bounds[0 + dirIsNeg[0]].x - ray.o.x) * invDir.x;
    float tMax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
    float tyMin = (bounds[0 + dirIsNeg[1]].y - ray.o.y) * invDir.y;
    float tyMax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;
    if (tMin > tyMax || tyMin > tMax) return false;
    if (tyMin > tMin) tMin = tyMin;
    if (tyMax < tMax) tMax = tyMax;
    // Check for ray intersection against z slab
    float tzMin = (bounds[0 + dirIsNeg[2]].z - ray.o.z) * invDir.z;
    float tzMax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;
    if (tMin > tzMax || tzMin > tMax) return false;
    if (tzMin > tMin) tMin = tzMin;
    if (tzMax < tMax) tMax = tzMax;
    return (tMin < ray.tMax) && (tMax > 0);
}

SE_EXPORT template <class T>
inline auto unionPoint(Bounds3<T> const& b, Point3<T> const& p) noexcept
    -> Bounds3<T> {
    return Bounds3<T>(
        Point3<T>(std::min(b.pMin.x, p.x), std::min(b.pMin.y, p.y),
                  std::min(b.pMin.z, p.z)),
        Point3<T>(std::max(b.pMax.x, p.x), std::max(b.pMax.y, p.y),
                  std::max(b.pMax.z, p.z)));
}

SE_EXPORT template <class T>
inline auto unionBounds(Bounds3<T> const& b1, Bounds3<T> const& b2) noexcept
    -> Bounds3<T> {
    return Bounds3<T>(Point3<T>(std::min(b1.pMin.x, b2.pMin.x),
                                std::min(b1.pMin.y, b2.pMin.y),
                                std::min(b1.pMin.z, b2.pMin.z)),
                      Point3<T>(std::max(b1.pMax.x, b2.pMax.x),
                                std::max(b1.pMax.y, b2.pMax.y),
                                std::max(b1.pMax.z, b2.pMax.z)));
}

SE_EXPORT template <class T>
inline auto intersect(Bounds3<T> const& b1, Bounds3<T> const& b2) noexcept
    -> Bounds3<T> {
    return Bounds3<T>(Point3<T>(std::max(b1.pMin.x, b2.pMin.x),
                                std::max(b1.pMin.y, b2.pMin.y),
                                std::max(b1.pMin.z, b2.pMin.z)),
                      Point3<T>(std::min(b1.pMax.x, b2.pMax.x),
                                std::min(b1.pMax.y, b2.pMax.y),
                                std::min(b1.pMax.z, b2.pMax.z)));
}

SE_EXPORT template <class T>
inline auto overlaps(Bounds3<T> const& b1, Bounds3<T> const& b2) noexcept
    -> Bounds3<T> {
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
    return (x && y && z);
}

SE_EXPORT template <class T>
inline auto inside(Point3<T> const& p, Bounds3<T> const& b) noexcept -> bool {
    return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
            p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
}

SE_EXPORT template <class T>
inline auto insideExclusive(Point3<T> const& p, Bounds3<T> const& b) noexcept
    -> bool {
    return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y &&
            p.y < b.pMax.y && p.z >= b.pMin.z && p.z < b.pMax.z);
}

SE_EXPORT template <class T>
inline auto expand(Bounds3<T> const& b, float delta) noexcept -> bool {
    return Bounds3<T>(b.pMin - Vector3<T>(delta), b.pMax + Vector3<T>(delta));
}

SE_EXPORT using bounds3 = Bounds3<float>;
SE_EXPORT using ibounds3 = Bounds3<int32_t>;
}  // namespace SIByL::Math


namespace SIByL::Math {
SE_EXPORT inline auto sphericalDirection(float sinTheta, float cosTheta,
                                      float phi) noexcept -> Math::vec3 {
    return Math::vec3(sinTheta * std::cos(phi), sinTheta * std::sin(phi),
                      cosTheta);
}

SE_EXPORT inline auto sphericalDirection(float sinTheta, float cosTheta, float phi,
                                      Math::vec3 const& x, Math::vec3 const& y,
                                      Math::vec3 const& z) noexcept
    -> Math::vec3 {
    return sinTheta * std::cos(phi) * x + sinTheta * std::sin(phi) * y +
           cosTheta * z;
}

SE_EXPORT inline auto sphericalTheta(Math::vec3 const& v) noexcept -> float {
    return std::acos(Math::clamp(v.z, -1.f, 1.f));
}

SE_EXPORT inline auto sphericalPhi(Math::vec3 const& v) noexcept -> float {
    float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * Math::float_Pi) : p;
}
}  // namespace SIByL::Math

namespace SIByL::Math {
SE_EXPORT struct Quaternion {
    Quaternion() : v(0), s(1.f) {}
    Quaternion(float x, float y, float z, float w) : v(x, y, z), s(w) {}
    Quaternion(vec3 const& v, float s) : v(v), s(s) {}
    Quaternion(vec3 const& eulerAngle);
    Quaternion(mat3 const& m);
    Quaternion(mat4 const& m);

    auto toMat3() const noexcept -> mat3;
    auto toMat4() const noexcept -> mat4;

    auto lengthSquared() const -> float;
    auto length() const -> float;

    auto conjugate() noexcept -> Quaternion;
    auto reciprocal() noexcept -> Quaternion;

    auto operator/(float s) const -> Quaternion;
    auto operator+(Quaternion const& q2) const -> Quaternion;
    auto operator*(Quaternion const& q2) const -> Quaternion;
    auto operator*(Math::vec3 const& v) const -> Math::vec3;
    auto operator+=(Quaternion const& q) -> Quaternion&;
    auto operator-() const -> Quaternion;

    union {
    struct {
      vec3 v;
      float s;
    };
    float data[4];
    struct {
      float x, y, z, w;
    };
    };
};

SE_EXPORT inline auto dot(Quaternion const& q1, Quaternion const& q2) noexcept
    -> float {
    return dot(q1.v, q2.v) + q1.s * q2.s;
}

SE_EXPORT inline auto normalize(Quaternion const& q) noexcept -> Quaternion {
    return q / std::sqrt(dot(q, q));
}

SE_EXPORT inline auto operator*(float s, Quaternion const& q)->Quaternion {
    return Quaternion{s * q.x, s * q.y, s * q.z, s * q.w};
}

SE_EXPORT inline auto operator*(Quaternion const& q, float s)->Quaternion {
    return Quaternion{s * q.x, s * q.y, s * q.z, s * q.w};
}

SE_EXPORT inline auto operator-(Quaternion const& q1, Quaternion const& q2)
    -> Quaternion {
    return Quaternion{q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.w - q2.w};
}

SE_EXPORT inline auto slerp(float t, Quaternion const& q1,
                            Quaternion const& q2) noexcept -> Quaternion {
  Quaternion previousQuat=q1, nextQuat = q2;
  float dotProduct = Math::dot(previousQuat, nextQuat);
  // make sure we take the shortest path in case dot Product is negative
  if (dotProduct < 0.0) {
    nextQuat = -nextQuat;
    dotProduct = -dotProduct;
  }
  // if the two quaternions are too close to each other, just linear
  // interpolate between the 4D vector
  if (dotProduct > 0.9995) {
    return Math::normalize(previousQuat + t * (nextQuat - previousQuat));
  }
  // perform the spherical linear interpolation
  float theta_0 = std::acos(dotProduct);
  float theta = t * theta_0;
  float sin_theta = std::sin(theta);
  float sin_theta_0 = std::sin(theta_0);
  float scalePreviousQuat = std::cos(theta) - dotProduct * sin_theta / sin_theta_0;
  float scaleNextQuat = sin_theta / sin_theta_0;
  return scalePreviousQuat * previousQuat + scaleNextQuat * nextQuat;
}

SE_EXPORT inline auto offsetRayOrigin(Math::point3 const& p,
                                      Math::vec3 const& pError,
                                      Math::normal3 const& n,
                                      Math::vec3 const& w) noexcept
    -> Math::point3 {
    float d = dot((vec3)abs(n), pError);
    Math::vec3 offset = d * Math::vec3(n);
    if (dot(w, n) < 0) offset = -offset;
    Math::point3 po = p + offset;
    // Round offset point po away from p
    for (int i = 0; i < 3; ++i) {
    if (offset.at(i) > 0)
      po.at(i) = nextFloatUp(po.at(i));
    else if (offset.at(i) < 0)
      po.at(i) = nextFloatDown(po.at(i));
    }
    return po;
}
}  // namespace SIByL::Math

namespace SIByL::Math {
SE_EXPORT struct Transform {
    Transform() = default;
    Transform(float const mat[4][4]);
    Transform(mat4 const& m);
    Transform(mat4 const& m, mat4 const& mInverse);
    Transform(Quaternion const& q);

    auto isIdentity() const noexcept -> bool;
    auto hasScale() const noexcept -> bool;
    auto swapsHandness() const noexcept -> bool;

    auto operator==(Transform const& t) const -> bool;
    auto operator!=(Transform const& t) const -> bool;

    auto operator*(point3 const& p) const -> point3;
    auto operator*(vec3 const& v) const -> vec3;
    auto operator*(normal3 const& n) const -> normal3;
    auto operator*(ray3 const& s) const -> ray3;
    auto operator*(bounds3 const& b) const -> bounds3;
    auto operator*(Transform const& t2) const -> Transform;

    auto operator()(point3 const& p, vec3& absError) const -> point3;
    auto operator()(point3 const& p, vec3 const& pError, vec3& tError) const
        -> point3;
    auto operator()(vec3 const& v, vec3& absError) const -> vec3;
    auto operator()(vec3 const& v, vec3 const& pError, vec3& tError) const
        -> vec3;
    auto operator()(ray3 const& r, vec3& oError, vec3& dError) const -> ray3;

    friend auto inverse(Transform const& t) noexcept -> Transform;
    friend auto transpose(Transform const& t) noexcept -> Transform;

    mat4 m;
    mat4 mInv;
};

SE_EXPORT inline auto inverse(Transform const& t) noexcept -> Transform;
SE_EXPORT inline auto transpose(Transform const& t) noexcept -> Transform;

SE_EXPORT inline auto translate(vec3 const& delta) noexcept -> Transform;
SE_EXPORT inline auto scale(float x, float y, float z) noexcept -> Transform;
SE_EXPORT inline auto rotateX(float theta) noexcept -> Transform;
SE_EXPORT inline auto rotateY(float theta) noexcept -> Transform;
SE_EXPORT inline auto rotateZ(float theta) noexcept -> Transform;
SE_EXPORT inline auto rotate(float theta, vec3 const& axis) noexcept -> Transform;

SE_EXPORT inline auto lookAt(point3 const& pos, point3 const& look,
                          vec3 const& up) noexcept -> Transform;

SE_EXPORT inline auto orthographic(float zNear, float zFar) noexcept -> Transform;
SE_EXPORT inline auto ortho(float left, float right, float bottom, float top,
                         float zNear, float zFar) noexcept -> Transform;

SE_EXPORT inline auto perspective(float fov, float n, float f) noexcept
    -> Transform;
SE_EXPORT inline auto perspective(float fov, float aspect, float n,
                               float f) noexcept -> Transform;

/** Decompose an affine transformation into Translation x Rotation x Scaling */
SE_EXPORT inline auto decompose(mat4 const& m, vec3* t, Quaternion* rquat,
                             mat4* s) noexcept -> void;
SE_EXPORT inline auto decompose(mat4 const& m, vec3* t, vec3* r, vec3* s) noexcept
    -> void;

}  // namespace SIByL::Math

namespace SIByL::Math {
    using namespace SIByL::Math;

    /** Light weight ray triangle intersection
    * @ref: https://github.com/SebLague/Gamedev-Maths/blob/master/PointInTriangle.cs */
    SE_EXPORT inline auto ray_triangle_intersection_lightweight(
        vec3 const& ray, vec3 const& rayDir,
        vec3 const& A, vec3 const& B, vec3 const& C
    ) noexcept -> bool {
        vec3 normal = normalize(cross(B - A, C - A));
        float t = dot(-ray, normal) / dot(normal, rayDir);
        vec3 P = ray + rayDir * t;  // hit point

        float s1 = C.y - A.y;
        float s2 = C.x - A.x;
        float s3 = B.y - A.y;
        float s4 = P.y - A.y;

        float w1 = (A.x * s1 + s4 * s2 - P.x * s1) / (s3 * s2 - (B.x - A.x) * s1);
        float w2 = (s4 - w1 * s3) / s1;
        return w1 >= 0.0 && w2 >= 0.0 && (w1 + w2) <= 1.0 ? true : false;
    }
}

namespace SIByL::Math {

inline auto inverse(Transform const& t) noexcept -> Transform {
    return Transform(t.mInv, t.m);
}

inline auto transpose(Transform const& t) noexcept -> Transform {
    return Transform(transpose(t.m), transpose(t.mInv));
}

inline auto translate(vec3 const& delta) noexcept -> Transform {
    mat4 m(1, 0, 0, delta.x, 0, 1, 0, delta.y, 0, 0, 1, delta.z, 0, 0, 0, 1);

    mat4 minv(1, 0, 0, -delta.x, 0, 1, 0, -delta.y, 0, 0, 1, -delta.z, 0, 0, 0,
              1);

    return Transform(m, minv);
}

inline auto scale(float x, float y, float z) noexcept -> Transform {
    mat4 m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);

    mat4 minv(1.f / x, 0, 0, 0, 0, 1.f / y, 0, 0, 0, 0, 1.f / z, 0, 0, 0, 0, 1);

    return Transform(m, minv);
}

inline auto rotateX(float theta) noexcept -> Transform {
    float sinTheta = std::sin(Math::radians(theta));
    float cosTheta = std::cos(Math::radians(theta));
    mat4 m(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0,
           0, 0, 1);
    return Transform(m, transpose(m));
}

inline auto rotateY(float theta) noexcept -> Transform {
    float sinTheta = std::sin(Math::radians(theta));
    float cosTheta = std::cos(Math::radians(theta));
    mat4 m(cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0, 0,
           0, 0, 1);
    return Transform(m, transpose(m));
}

inline auto rotateZ(float theta) noexcept -> Transform {
    float sinTheta = std::sin(Math::radians(theta));
    float cosTheta = std::cos(Math::radians(theta));
    mat4 m(cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 1);
    return Transform(m, transpose(m));
}

inline auto rotate(float theta, vec3 const& axis) noexcept -> Transform {
    vec3 a = normalize(axis);
    float sinTheta = std::sin(Math::radians(theta));
    float cosTheta = std::cos(Math::radians(theta));
    mat4 m;
    // Compute rotation of first basis vector
    m.data[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m.data[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m.data[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m.data[0][3] = 0;
    // Compute rotations of second and third basis vectors
    m.data[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m.data[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m.data[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m.data[1][3] = 0;

    m.data[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m.data[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m.data[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m.data[2][3] = 0;

    return Transform(m, transpose(m));
}

inline auto lookAt(point3 const& pos, point3 const& look,
                   vec3 const& up) noexcept -> Transform {
    mat4 cameraToWorld;
    // Initialize fourth column of viewing matrix
    cameraToWorld.data[0][3] = pos.x;
    cameraToWorld.data[1][3] = pos.y;
    cameraToWorld.data[2][3] = pos.z;
    cameraToWorld.data[3][3] = 1;
    // Initialize first three columns of viewing matrix
    vec3 dir = normalize(look - pos);
    vec3 left = normalize(cross(dir, normalize(up)));
    vec3 newUp = cross(dir, left);
    cameraToWorld.data[0][0] = left.x;
    cameraToWorld.data[0][1] = newUp.x;
    cameraToWorld.data[0][2] = dir.x;
    cameraToWorld.data[1][0] = left.y;
    cameraToWorld.data[1][1] = newUp.y;
    cameraToWorld.data[1][2] = dir.y;
    cameraToWorld.data[2][0] = left.z;
    cameraToWorld.data[2][1] = newUp.z;
    cameraToWorld.data[2][2] = dir.z;
    cameraToWorld.data[3][0] = 0;
    cameraToWorld.data[3][1] = 0;
    cameraToWorld.data[3][2] = 0;
    return Transform(inverse(cameraToWorld), cameraToWorld);
}

inline auto orthographic(float zNear, float zFar) noexcept -> Transform {
    return Math::scale(1, 1, 1.f / (zFar - zNear)) *
           Math::translate({0, 0, -zNear});
}

inline auto ortho(float left, float right, float bottom, float top, float zNear,
                  float zFar) noexcept -> Transform {
    Math::mat4 trans = {float(2) / (right - left),
                        0,
                        0,
                        -(right + left) / (right - left),
                        0,
                        float(2) / (top - bottom),
                        0,
                        -(top + bottom) / (top - bottom),
                        0,
                        0,
                        float(1) / (zFar - zNear),
                        -1 * zNear / (zFar - zNear),
                        0,
                        0,
                        0,
                        1};
    return Transform(trans);
}

inline auto perspective(float fov, float n, float f) noexcept -> Transform {
    // perform projective divide for perspective projection
    mat4 persp{1.0f, 0.0f, 0.0f, 0.0f, 0.0f,        1.0f,
               0.0f, 0.0f, 0.0f, 0.0f, f / (f - n), -f * n / (f - n),
               0.0f, 0.0f, 1.0f, 0.0f};
    // scale canonical perspective view to specified field of view
    float invTanAng = 1.f / std::tan(radians(fov) / 2);
    return scale(invTanAng, invTanAng, 1) * Transform(persp);
}

inline auto perspective(float fov, float aspect, float n, float f) noexcept
    -> Transform {
    // perform projective divide for perspective projection
    mat4 persp{1.0f, 0.0f, 0.0f, 0.0f, 0.0f,        1.0f,
               0.0f, 0.0f, 0.0f, 0.0f, f / (f - n), -f * n / (f - n),
               0.0f, 0.0f, 1.0f, 0.0f};
    // scale canonical perspective view to specified field of view
    float invTanAng = 1.f / std::tan(radians(fov) / 2);
    return scale(invTanAng / aspect, invTanAng, 1) * Transform(persp);
}

inline auto decompose(mat4 const& m, vec3* t, Quaternion* rquat,
                      mat4* s) noexcept -> void {
    // Extract translation T from transformation matrix
    // which could be found directly from matrix
    t->x = m.data[0][3];
    t->y = m.data[1][3];
    t->z = m.data[2][3];

    // Compute new transformation matrix M without translation
    mat4 M = m;
    for (int i = 0; i < 3; i++) M.data[i][3] = M.data[3][i] = 0.f;
    M.data[3][3] = 1.f;

    // Extract rotation R from transformation matrix
    // use polar decomposition, decompose into R&S by averaging M with its
    // inverse transpose until convergence to get R (because pure rotation
    // matrix has similar inverse and transpose)
    float norm;
    int count = 0;
    mat4 R = M;
    do {
    // Compute next matrix Rnext in series
    mat4 rNext;
    mat4 rInvTrans = inverse(transpose(R));
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        rNext.data[i][j] = 0.5f * (R.data[i][j] + rInvTrans.data[i][j]);
    // Compute norm of difference between R and Rnext
    norm = 0.f;
    for (int i = 0; i < 3; ++i) {
      float n = std::abs(R.data[i][0] = rNext.data[i][0]) +
                std::abs(R.data[i][1] = rNext.data[i][1]) +
                std::abs(R.data[i][2] = rNext.data[i][2]);
      norm = std::max(norm, n);
    }
    R = rNext;
    } while (++count < 100 && norm > .0001);
    *rquat = Quaternion(R);
    // Compute scale S using rotationand original matrix
    *s = mul(inverse(R), M);
}

inline auto decompose(mat4 const& m, vec3* t, vec3* r, vec3* s) noexcept
    -> void {
    // Extract translation T from transformation matrix
    // which could be found directly from matrix
    t->x = m.data[0][3];
    t->y = m.data[1][3];
    t->z = m.data[2][3];

    // Compute new transformation matrix M without translation
    mat4 M = m;
    for (int i = 0; i < 3; i++) M.data[i][3] = M.data[3][i] = 0.f;
    M.data[3][3] = 1.f;

    // Extract rotation R from transformation matrix
    // use polar decomposition, decompose into R&S by averaging M with its
    // inverse transpose until convergence to get R (because pure rotation
    // matrix has similar inverse and transpose)
    float norm;
    int count = 0;
    mat4 R = M;
    do {
    // Compute next matrix Rnext in series
    mat4 rNext;
    mat4 rInvTrans = inverse(transpose(R));
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        rNext.data[i][j] = 0.5f * (R.data[i][j] + rInvTrans.data[i][j]);
    // Compute norm of difference between R and Rnext
    norm = 0.f;
    for (int i = 0; i < 3; ++i) {
      float n = std::abs(R.data[i][0] = rNext.data[i][0]) +
                std::abs(R.data[i][1] = rNext.data[i][1]) +
                std::abs(R.data[i][2] = rNext.data[i][2]);
      norm = std::max(norm, n);
    }
    R = rNext;
    } while (++count < 100 && norm > .0001);

    r->y = std::asin(-R.data[2][0]);
    if (std::cos(r->y) != 0) {
    r->x = atan2(R.data[2][1], R.data[2][2]);
    r->z = atan2(R.data[1][0], R.data[0][0]);
    } else {
    r->x = atan2(-R.data[0][2], R.data[1][1]);
    r->z = 0;
    }

    // Compute scale S using rotationand original matrix
    mat4 smat = mul(inverse(R), M);
    s->x =
        Math::vec3(smat.data[0][0], smat.data[1][0], smat.data[2][0]).length();
    s->y =
        Math::vec3(smat.data[0][1], smat.data[1][1], smat.data[2][1]).length();
    s->z =
        Math::vec3(smat.data[0][2], smat.data[1][2], smat.data[2][2]).length();
}


}