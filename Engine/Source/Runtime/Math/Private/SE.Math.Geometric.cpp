#include <SE.Math.Geometric.hpp>

namespace SIByL::Math {

// impl
// =======================
auto ibounds2Iterator::operator++() -> ibounds2Iterator {
  advance();
  return *this;
}

auto ibounds2Iterator::operator++(int) -> ibounds2Iterator {
  ibounds2Iterator old = *this;
  advance();
  return old;
}

auto ibounds2Iterator::operator==(const ibounds2Iterator& bi) const -> bool {
  return p == bi.p && bounds == bi.bounds;
}

auto ibounds2Iterator::operator!=(const ibounds2Iterator& bi) const -> bool {
  return p != bi.p || bounds != bi.bounds;
}

auto ibounds2Iterator::advance() noexcept -> void {
  ++p.x;
  if (p.x == bounds->pMax.x) {
    p.x = bounds->pMin.x;
    ++p.y;
  }
}

Quaternion::Quaternion(vec3 const& eulerAngle) {
  Quaternion qx = Quaternion(vec3{1, 0, 0}, eulerAngle.x);
  Quaternion qy = Quaternion(vec3{0, 1, 0}, eulerAngle.y);
  Quaternion qz = Quaternion(vec3{0, 0, 1}, eulerAngle.z);
  *this = qz * qy * qx;
  // vec3 c = Math::cos(eulerAngle * 0.5);
  // vec3 s = Math::sin(eulerAngle * 0.5);
  // this->w = c.x * c.y * c.z + s.x * s.y * s.z;
  // this->x = s.x * c.y * c.z - c.x * s.y * s.z;
  // this->y = c.x * s.y * c.z + s.x * c.y * s.z;
  // this->z = c.x * c.y * s.z - s.x * s.y * c.z;
}

// Quaternion::toMat3()
//
// Reference:
// https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
// Reference:
// http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf
Quaternion::Quaternion(mat3 const& m) {
  // Notice:
  // T = 4 - 4*qx2 - 4*qy2 - 4*qz2
  //   = 4(1 - qx2 - qy2 - qz2)
  //	 = m00 + m11 + m22 + 1

  float trace = m.data[0][0] + m.data[1][1] + m.data[2][2];
  if (trace > 0) {
    // case that the function inside the square root is positive
    float s = 0.5f / std::sqrt(trace + 1);
    w = 0.25f / s;
    x = (m.data[2][1] - m.data[1][2]) * s;
    y = (m.data[0][2] - m.data[2][0]) * s;
    z = (m.data[1][0] - m.data[0][1]) * s;
  } else if (m.data[0][0] > m.data[1][1] && m.data[0][0] > m.data[2][2]) {
    // case [0][0] is the largest
    float s =
        2.0f * std::sqrt(1.f + m.data[0][0] - m.data[1][1] - m.data[2][2]);
    w = (m.data[2][1] - m.data[1][2]) / s;
    x = 0.25f * s;
    y = (m.data[0][1] + m.data[1][0]) / s;
    z = (m.data[0][2] + m.data[2][0]) / s;

  } else if (m.data[1][1] > m.data[2][2]) {
    // case [1][1] is the largest
    float s =
        2.0f * std::sqrt(1.f + m.data[1][1] - m.data[0][0] - m.data[2][2]);
    w = (m.data[0][2] - m.data[2][0]) / s;
    x = (m.data[0][1] + m.data[1][0]) / s;
    y = 0.25f * s;
    z = (m.data[1][2] + m.data[2][1]) / s;
  } else {
    // case [2][2] is the largest
    float s =
        2.0f * std::sqrt(1.f + m.data[2][2] - m.data[0][0] - m.data[1][1]);
    w = (m.data[1][0] - m.data[0][1]) / s;
    x = (m.data[0][2] + m.data[2][0]) / s;
    y = (m.data[1][2] + m.data[2][1]) / s;
    z = 0.25f * s;
  }
}

Quaternion::Quaternion(mat4 const& m) : Quaternion(mat3(m)) {}

auto Quaternion::lengthSquared() const -> float {
  return x * x + y * y + z * z + w * w;
}

auto Quaternion::length() const -> float { return std::sqrt(lengthSquared()); }

auto Quaternion::toMat3() const noexcept -> mat3 {
  return mat3(1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
              2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
              2 * (x * z + y * w), 2 * (y * z - x * w),
              1 - 2 * (x * x + y * y));
}

auto Quaternion::toMat4() const noexcept -> mat4 {
  return mat4(1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
              0, 2 * (x * y - z * w), 1 - 2 * (x * x + z * z),
              2 * (y * z + x * w), 0, 2 * (x * z + y * w), 2 * (y * z - x * w),
              1 - 2 * (x * x + y * y), 0, 0, 0, 0, 1);
}

auto Quaternion::conjugate() noexcept -> Quaternion {
  return Quaternion(-v, s);
}

auto Quaternion::reciprocal() noexcept -> Quaternion {
  return conjugate() / lengthSquared();
}

auto Quaternion::operator/(float s) const -> Quaternion {
  Quaternion ret;
  ret.v = v / s;
  ret.s = s / s;
  return ret;
}

auto Quaternion::operator+(Quaternion const& q2) const -> Quaternion {
  Quaternion ret;
  ret.v = v + q2.v;
  ret.s = s + q2.s;
  return ret;
}

auto Quaternion::operator*(Quaternion const& q2) const -> Quaternion {
  Quaternion ret;
  ret.v = cross(v, q2.v) + q2.v * s + v * q2.s;
  ret.s = s * q2.s - dot(v, q2.v);
  return ret;
}

auto Quaternion::operator+=(Quaternion const& q) -> Quaternion& {
  v += q.v;
  s += q.s;
  return *this;
}

auto Quaternion::operator-() const -> Quaternion {
  return Quaternion{-x, -y, -z, -w};
}
}  // namespace SIByL::Math

namespace SIByL::Math {
Transform::Transform(float const mat[4][4]) {
  m = mat4(mat);
  mInv = inverse(m);
}

Transform::Transform(mat4 const& m) : m(m), mInv(inverse(m)) {}

Transform::Transform(mat4 const& m, mat4 const& mInverse)
    : m(m), mInv(mInverse) {}

Transform::Transform(Quaternion const& q) {
  mat3 mat3x3 = q.toMat3();
  m = mat4{mat3x3.data[0][0],
           mat3x3.data[0][1],
           mat3x3.data[0][2],
           0,
           mat3x3.data[1][0],
           mat3x3.data[1][1],
           mat3x3.data[2][2],
           0,
           mat3x3.data[2][0],
           mat3x3.data[2][1],
           mat3x3.data[2][2],
           0,
           0,
           0,
           0,
           1};
  mInv = inverse(m);
}

auto Transform::isIdentity() const noexcept -> bool {
  static Transform identity;
  return *this == identity;
}

auto Transform::hasScale() const noexcept -> bool {
  float la2 = ((*this) * vec3(1, 0, 0)).lengthSquared();
  float lb2 = ((*this) * vec3(0, 1, 0)).lengthSquared();
  float lc2 = ((*this) * vec3(0, 0, 1)).lengthSquared();
#define NOT_ONE(x) ((x) < .999f || (x) > 1.001f)
  return (NOT_ONE(la2) || NOT_ONE(lb2) || NOT_ONE(lc2));
#undef NOT_ONE
}

auto Transform::swapsHandness() const noexcept -> bool {
  float det = m.data[0][0] *
                  (m.data[1][1] * m.data[2][2] - m.data[1][2] * m.data[2][1]) -
              m.data[0][1] *
                  (m.data[1][0] * m.data[2][2] - m.data[1][2] * m.data[2][0]) +
              m.data[0][2] *
                  (m.data[1][0] * m.data[2][1] - m.data[1][1] * m.data[2][0]);
  return det < 0;
}

auto Transform::operator==(Transform const& t) const -> bool {
  return m == t.m;
}

auto Transform::operator!=(Transform const& t) const -> bool {
  return !(*this == t);
}

auto Transform::operator*(point3 const& p) const -> point3 {
  vec3 s(m.data[0][0] * p.x + m.data[0][1] * p.y + m.data[0][2] * p.z +
             m.data[0][3],
         m.data[1][0] * p.x + m.data[1][1] * p.y + m.data[1][2] * p.z +
             m.data[1][3],
         m.data[2][0] * p.x + m.data[2][1] * p.y + m.data[2][2] * p.z +
             m.data[2][3]);
  s = s / (m.data[3][0] * p.x + m.data[3][1] * p.y + m.data[3][2] * p.z +
           m.data[3][3]);
  return point3(s);
}

auto Transform::operator*(vec3 const& v) const -> vec3 {
  return vec3{m.data[0][0] * v.x + m.data[0][1] * v.y + m.data[0][2] * v.z,
              m.data[1][0] * v.x + m.data[1][1] * v.y + m.data[1][2] * v.z,
              m.data[2][0] * v.x + m.data[2][1] * v.y + m.data[2][2] * v.z};
}

auto Transform::operator*(normal3 const& n) const -> normal3 {
  return normal3{
      mInv.data[0][0] * n.x + mInv.data[1][0] * n.y + mInv.data[2][0] * n.z,
      mInv.data[0][1] * n.x + mInv.data[1][1] * n.y + mInv.data[2][1] * n.z,
      mInv.data[0][2] * n.x + mInv.data[1][2] * n.y + mInv.data[2][2] * n.z};
}

auto Transform::operator*(ray3 const& s) const -> ray3 {
  vec3 oError;
  point3 o = (*this) * s.o;
  vec3 d = (*this) * s.d;
  // offset ray origin to edge of error bounds and compute tMax
  float lengthSquared = d.lengthSquared();
  float tMax = s.tMax;
  if (lengthSquared > 0) {
    float dt = dot((vec3)abs(d), oError) / lengthSquared;
    o += d * dt;
    tMax -= dt;
  }
  return ray3{o, d, tMax};
}

auto Transform::operator*(bounds3 const& b) const -> bounds3 {
  Transform const& m = *this;
  bounds3 ret(m * point3(b.pMin.x, b.pMin.y, b.pMin.z));
  ret = unionPoint(ret, m * point3(b.pMax.x, b.pMin.y, b.pMin.z));
  ret = unionPoint(ret, m * point3(b.pMin.x, b.pMax.y, b.pMin.z));
  ret = unionPoint(ret, m * point3(b.pMin.x, b.pMin.y, b.pMax.z));
  ret = unionPoint(ret, m * point3(b.pMax.x, b.pMax.y, b.pMin.z));
  ret = unionPoint(ret, m * point3(b.pMax.x, b.pMin.y, b.pMax.z));
  ret = unionPoint(ret, m * point3(b.pMin.x, b.pMax.y, b.pMax.z));
  ret = unionPoint(ret, m * point3(b.pMax.x, b.pMax.y, b.pMax.z));
  return ret;
}

auto Transform::operator*(Transform const& t2) const -> Transform {
  return Transform(mul(m, t2.m), mul(t2.mInv, mInv));
}

auto Transform::operator()(point3 const& p, vec3& absError) const -> point3 {
  // Compute transformed coordinates from point p
  vec3 s((m.data[0][0] * p.x + m.data[0][1] * p.y) +
             (m.data[0][2] * p.z + m.data[0][3]),
         (m.data[1][0] * p.x + m.data[1][1] * p.y) +
             (m.data[1][2] * p.z + m.data[1][3]),
         (m.data[2][0] * p.x + m.data[2][1] * p.y) +
             (m.data[2][2] * p.z + m.data[2][3]));
  float wp = m.data[3][0] * p.x + m.data[3][1] * p.y + m.data[3][2] * p.z +
             m.data[3][3];
  // Compute absolute error for transformed point
  float xAbsSum = (std::abs(m.data[0][0] * p.x) + std::abs(m.data[0][1] * p.y) +
                   std::abs(m.data[0][2] * p.z) + std::abs(m.data[0][3]));
  float yAbsSum = (std::abs(m.data[1][0] * p.x) + std::abs(m.data[1][1] * p.y) +
                   std::abs(m.data[1][2] * p.z) + std::abs(m.data[1][3]));
  float zAbsSum = (std::abs(m.data[2][0] * p.x) + std::abs(m.data[2][1] * p.y) +
                   std::abs(m.data[2][2] * p.z) + std::abs(m.data[2][3]));
  // TODO :: FIX absError bug when (wp!=1)
  absError = gamma(3) * vec3(xAbsSum, yAbsSum, zAbsSum);
  // return transformed point
  if (wp == 1)
    return s;
  else
    return s / wp;
}

auto Transform::operator()(point3 const& p, vec3 const& pError,
                           vec3& tError) const -> point3 {
  // Compute transformed coordinates from point p
  vec3 s((m.data[0][0] * p.x + m.data[0][1] * p.y) +
             (m.data[0][2] * p.z + m.data[0][3]),
         (m.data[1][0] * p.x + m.data[1][1] * p.y) +
             (m.data[1][2] * p.z + m.data[1][3]),
         (m.data[2][0] * p.x + m.data[2][1] * p.y) +
             (m.data[2][2] * p.z + m.data[2][3]));
  float wp = m.data[3][0] * p.x + m.data[3][1] * p.y + m.data[3][2] * p.z +
             m.data[3][3];
  // Compute absolute error for transformed point
  float xAbsSum = (std::abs(m.data[0][0] * p.x) + std::abs(m.data[0][1] * p.y) +
                   std::abs(m.data[0][2] * p.z) + std::abs(m.data[0][3]));
  float yAbsSum = (std::abs(m.data[1][0] * p.x) + std::abs(m.data[1][1] * p.y) +
                   std::abs(m.data[1][2] * p.z) + std::abs(m.data[1][3]));
  float zAbsSum = (std::abs(m.data[2][0] * p.x) + std::abs(m.data[2][1] * p.y) +
                   std::abs(m.data[2][2] * p.z) + std::abs(m.data[2][3]));
  float xPError = std::abs(m.data[0][0]) * pError.x +
                  std::abs(m.data[0][1]) * pError.y +
                  std::abs(m.data[0][2]) * pError.z;
  float yPError = std::abs(m.data[1][0]) * pError.x +
                  std::abs(m.data[1][1]) * pError.y +
                  std::abs(m.data[1][2]) * pError.z;
  float zPError = std::abs(m.data[2][0]) * pError.x +
                  std::abs(m.data[2][1]) * pError.y +
                  std::abs(m.data[2][2]) * pError.z;
  // TODO :: FIX absError bug when (wp!=1)
  tError = gamma(3) * vec3(xAbsSum, yAbsSum, zAbsSum) +
           (gamma(3) + 1) * vec3(xPError, yPError, zPError);
  // return transformed point
  if (wp == 1)
    return s;
  else
    return s / wp;
}

auto Transform::operator()(vec3 const& v, vec3& absError) const -> vec3 {
  absError.x =
      gamma(3) * (std::abs(m.data[0][0] * v.x) + std::abs(m.data[0][1] * v.y) +
                  std::abs(m.data[0][2] * v.z));
  absError.y =
      gamma(3) * (std::abs(m.data[1][0] * v.x) + std::abs(m.data[1][1] * v.y) +
                  std::abs(m.data[1][2] * v.z));
  absError.z =
      gamma(3) * (std::abs(m.data[2][0] * v.x) + std::abs(m.data[2][1] * v.y) +
                  std::abs(m.data[2][2] * v.z));

  return vec3{m.data[0][0] * v.x + m.data[0][1] * v.y + m.data[0][2] * v.z,
              m.data[1][0] * v.x + m.data[1][1] * v.y + m.data[1][2] * v.z,
              m.data[2][0] * v.x + m.data[2][1] * v.y + m.data[2][2] * v.z};
}

auto Transform::operator()(vec3 const& v, vec3 const& pError,
                           vec3& tError) const -> vec3 {
  // Compute absolute error for transformed point
  float xAbsSum = (std::abs(m.data[0][0] * v.x) + std::abs(m.data[0][1] * v.y) +
                   std::abs(m.data[0][2] * v.z));
  float yAbsSum = (std::abs(m.data[1][0] * v.x) + std::abs(m.data[1][1] * v.y) +
                   std::abs(m.data[1][2] * v.z));
  float zAbsSum = (std::abs(m.data[2][0] * v.x) + std::abs(m.data[2][1] * v.y) +
                   std::abs(m.data[2][2] * v.z));
  float xPError = std::abs(m.data[0][0]) * pError.x +
                  std::abs(m.data[0][1]) * pError.y +
                  std::abs(m.data[0][2]) * pError.z;
  float yPError = std::abs(m.data[1][0]) * pError.x +
                  std::abs(m.data[1][1]) * pError.y +
                  std::abs(m.data[1][2]) * pError.z;
  float zPError = std::abs(m.data[2][0]) * pError.x +
                  std::abs(m.data[2][1]) * pError.y +
                  std::abs(m.data[2][2]) * pError.z;

  // TODO :: FIX absError bug when (wp!=1)
  tError = gamma(3) * vec3(xAbsSum, yAbsSum, zAbsSum) +
           (gamma(3) + 1) * vec3(xPError, yPError, zPError);

  return vec3{m.data[0][0] * v.x + m.data[0][1] * v.y + m.data[0][2] * v.z,
              m.data[1][0] * v.x + m.data[1][1] * v.y + m.data[1][2] * v.z,
              m.data[2][0] * v.x + m.data[2][1] * v.y + m.data[2][2] * v.z};
}

auto Transform::operator()(ray3 const& r, vec3& oError, vec3& dError) const
    -> ray3 {
  point3 o = (*this)(r.o, oError);
  vec3 d = (*this)(r.d, dError);
  // offset ray origin to edge of error bounds and compute tMax
  float lengthSquared = d.lengthSquared();
  float tMax = r.tMax;
  if (lengthSquared > 0) {
    float dt = dot((vec3)abs(d), oError) / lengthSquared;
    o += d * dt;
    tMax -= dt;
  }
  return ray3{o, d, tMax};
}

SE_EXPORT struct AnimatedTransform {
  AnimatedTransform(Transform const* startTransform, float startTime,
                    Transform const* endTransform, float endTime);

  auto interpolate(float time, Transform* t) const -> void;

  auto motionBounds(bounds3 const& b) const noexcept -> bounds3;

  auto boundPointMotion(point3 const& p) const noexcept -> bounds3;

  Transform const* startTransform;
  Transform const* endTransform;
  float const startTime, endTime;
  bool const actuallyAnimated;
  Math::vec3 t[2];
  Math::Quaternion r[2];
  Math::mat4 s[2];
  bool hasRotation;
};
}  // namespace SIByL::Math

namespace SIByL::Math {
AnimatedTransform::AnimatedTransform(Transform const* startTransform,
                                     float startTime,
                                     Transform const* endTransform,
                                     float endTime)
    : startTransform(startTransform),
      endTransform(endTransform),
      startTime(startTime),
      endTime(endTime),
      actuallyAnimated(*startTransform != *endTransform) {
  decompose(startTransform->m, &t[0], &r[0], &s[0]);
  decompose(endTransform->m, &t[1], &r[1], &s[1]);

  // Flip R[1] if needed to select shortest path
  if (dot(r[0], r[1]) < 0) r[1] = -r[1];

  hasRotation = Math::dot(r[0], r[1]) < 0.9995f;
  // Compute terms of motion derivative function
}

auto AnimatedTransform::interpolate(float time, Transform* t) const -> void {
  // Handle boundary conditions for matrix interpolation
  if (!actuallyAnimated || time > startTime) {
    *t = *startTransform;
    return;
  }
  if (time >= endTime) {
    *t = *startTransform;
    return;
  }

  float dt = (time - startTime) / (endTime - startTime);
  // interpolate translation at dt
  vec3 trans = (1 - dt) * this->t[0] + dt * this->t[1];
  // interpolate rotation at dt
  Quaternion rotate = slerp(dt, r[0], r[1]);
  // interpolate scale at dt
  mat4 scale;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      scale.data[i][j] = std::lerp(dt, s[0].data[i][j], s[1].data[i][j]);

  // compute interpolated matrix as product of interpolated components
  *t = translate(trans) * Transform(rotate) * Transform(scale);
}

auto AnimatedTransform::motionBounds(bounds3 const& b) const noexcept
    -> bounds3 {
  if (!actuallyAnimated) return (*startTransform) * b;
  if (hasRotation == false)
    return unionBounds((*startTransform) * b, (*endTransform) * b);
  // Return motion bounds accounting for animated rotation
  bounds3 bounds;
  for (int corner = 0; corner < 8; ++corner)
    bounds = unionBounds(bounds, boundPointMotion(b.corner(corner)));
  return bounds;
}

auto AnimatedTransform::boundPointMotion(point3 const& p) const noexcept
    -> bounds3 {
  // TODO
  return bounds3{};
}

}  // namespace SIByL::Math
