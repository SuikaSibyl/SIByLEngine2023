#define DLIB_EXPORT
#include <se.math.hpp>
#undef DLIB_EXPORT

namespace se {
union uif32 {
  uif32() : i(0) {}
  uif32(float f) : f(f) {}
  uif32(unsigned int i) : i(i){}

  float f;
  unsigned int i;
};

typedef uif32 uif;

float overflow() {
  volatile float f = 1e10;
  for (int i = 0; i < 10; ++i)
    f *= f; // this will overflow before the forloop terminates
  return f;
}

half::half(float f) {
	uif Entry;
	Entry.f = f;
	int i = (int)Entry.i;

	int s =  (i >> 16) & 0x00008000;
	int e = ((i >> 23) & 0x000000ff) - (127 - 15);
	int m =   i        & 0x007fffff;

	if(e <= 0) {
		if(e < -10) {
			hdata = s;
      return;
		}

		m = (m | 0x00800000) >> (1 - e);

		if(m & 0x00001000) 
			m += 0x00002000;

    hdata = (s | (m >> 13));
    return;
	}
	else if(e == 0xff - (127 - 15)) {
		if(m == 0) {
      hdata = (s | 0x7c00);
      return;
		}
		else {
			m >>= 13;
      hdata = (s | 0x7c00 | m | (m == 0));
      return;
		}
	}
	else {
		if(m &  0x00001000) {
			m += 0x00002000;
			if(m & 0x00800000) {
				m =  0;     // overflow in significand,
				e += 1;     // adjust exponent
			}
		}

		if (e > 30) {
			overflow();        // Cause a hardware floating point overflow;
      hdata = (s | 0x7c00);
      return;
			// if this returns, the half becomes an
		}   // infinity with the same sign as f.
    hdata = (s | (e << 10) | (m >> 13));
    return;
	}
}

float half::to_float() const {
  int s = (hdata >> 15) & 0x00000001;
  int e = (hdata >> 10) & 0x0000001f;
  int m = hdata & 0x000003ff;

  if (e == 0) {
    if (m == 0) {
      uif result;
      result.i = (unsigned int)(s << 31);
      return result.f;
    }
    else {
      while (!(m & 0x00000400)) {
        m <<= 1;
        e -= 1;
      }

      e += 1;
      m &= ~0x00000400;
    }
  }
  else if (e == 31) {
    if (m == 0) {
      uif result;
      result.i = (unsigned int)((s << 31) | 0x7f800000);
      return result.f;
    } else {
      uif result;
      result.i = (unsigned int)((s << 31) | 0x7f800000 | (m << 13));
      return result.f;
    }
  }

  e = e + (127 - 15);
  m = m << 13;

  uif Result;
  Result.i = (unsigned int)((s << 31) | (e << 23) | m);
  return Result.f;
}

Quaternion::Quaternion(mat3 const& m) {
  // Notice:
  // T = 4 - 4*qx2 - 4*qy2 - 4*qz2
  //   = 4(1 - qx2 - qy2 - qz2)
  //   = m00 + m11 + m22 + 1
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
  // Math::mat4 quat_transform = quat.toMat4();
  se::vec3 x = (*this) * se::vec3(1, 0, 0);
  se::vec3 y = (*this) * se::vec3(0, 1, 0);
  se::vec3 z = (*this) * se::vec3(0, 0, 1);
  // Extract the position of the transform
  return se::mat3(x.x, y.x, z.x,  // X basis (& Scale)
                  x.y, y.y, z.y,  // Y basis (& scale)
                  x.z, y.z, z.z); // Z basis (& scale)
}

auto Quaternion::toMat4() const noexcept -> mat4 {
  // Math::mat4 quat_transform = quat.toMat4();
  se::vec3 x = (*this) * se::vec3(1, 0, 0);
  se::vec3 y = (*this) * se::vec3(0, 1, 0);
  se::vec3 z = (*this) * se::vec3(0, 0, 1);
  // Extract the position of the transform
  return se::mat4(x.x, y.x, z.x, 0,  // X basis (& Scale)
                  x.y, y.y, z.y, 0,  // Y basis (& scale)
                  x.z, y.z, z.z, 0,  // Z basis (& scale)
                  0, 0, 0, 1);
}

auto Quaternion::conjugate() noexcept -> Quaternion {
  return Quaternion(-v, s);
}

auto Quaternion::reciprocal() noexcept -> Quaternion {
  return conjugate() / lengthSquared();
}

auto Quaternion::operator/(float s) const -> Quaternion {
  Quaternion ret;
  ret.v = this->v / s;
  ret.s = this->s / s;
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

auto Quaternion::operator*(se::vec3 const& v) const -> se::vec3 {
  return this->v * 2.0f * se::dot(this->v, v) +
         v * (this->s * this->s - se::dot(this->v, this->v)) +
         se::cross(this->v, v) * 2.0f * this->s;
}

auto Quaternion::operator+=(Quaternion const& q) -> Quaternion& {
  v += q.v;
  s += q.s;
  return *this;
}

auto Quaternion::operator-() const -> Quaternion {
  return Quaternion{-x, -y, -z, -w};
}

float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5f;
inline auto gamma(int n) noexcept -> float {
  return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

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

auto Transform::operator()(point3 const& p, vec3 const& pError, vec3& tError) const -> point3 {
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

auto Transform::operator()(vec3 const& v, vec3 const& pError, vec3& tError) const -> vec3 {
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

auto Transform::operator()(ray3 const& r, vec3& oError, vec3& dError) const -> ray3 {
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

auto eulerAngleToRotationMatrix(se::vec3 eulerAngles) noexcept -> se::mat3 {
  return se::mat4::rotateZ(eulerAngles.z) *
    se::mat4::rotateY(eulerAngles.y) *
    se::mat4::rotateX(eulerAngles.x);
}

auto eulerAngleToQuaternion(se::vec3 eulerAngles) noexcept -> se::Quaternion {
  se::mat3 const mat = eulerAngleToRotationMatrix(eulerAngles);
  return se::Quaternion(mat);
}

auto rotationMatrixToEulerAngles(se::mat3 R) noexcept -> se::vec3 {
  float sy = std::sqrt(R.data[0][0] * R.data[0][0] + R.data[1][0] * R.data[1][0]);
  bool singular = sy < 1e-6;
  float x, y, z;
  if (!singular) {
    x = atan2(R.data[2][1], R.data[2][2]);
    y = atan2(-R.data[2][0], sy);
    z = atan2(R.data[1][0], R.data[0][0]);
  } else {
    x = atan2(-R.data[1][2], R.data[1][1]);
    y = atan2(-R.data[2][0], sy);
    z = 0;
  }
  return {x, y, z};
}

auto decompose(se::mat4 const& m, se::vec3* t, se::Quaternion* rquat, se::vec3* s) noexcept -> void {
  // Extract translation T from transformation matrix
  // which could be found directly from matrix
  t->x = m.data[0][3];
  t->y = m.data[1][3];
  t->z = m.data[2][3];

  // Compute new transformation matrix M without translation
  se::mat4 M = m;
  for (int i = 0; i < 3; i++) M.data[i][3] = M.data[3][i] = 0.f;
  M.data[3][3] = 1.f;

  // Extract rotation R from transformation matrix
  // use polar decomposition, decompose into R&S by averaging M with its
  // inverse transpose until convergence to get R (because pure rotation
  // matrix has similar inverse and transpose)
  float norm;
  int count = 0;
  se::mat4 R = M;
  do {
    // Compute next matrix Rnext in series
    se::mat4 rNext;
    se::mat4 rInvTrans = inverse(transpose(R));
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
  
  se::vec3 r, euler = rotationMatrixToEulerAngles(se::mat3{ R });
  r.x = euler.x * 180. / se::double_Pi;
  r.y = euler.y * 180. / se::double_Pi;
  r.z = euler.z * 180. / se::double_Pi;
  *rquat = eulerAngleToQuaternion(r);

  // Compute scale S using rotationand original matrix
  se::mat4 invR = inverse(R);
  se::mat4 Mtmp = M;
  se::mat4 smat = mul(invR, Mtmp);
  s->x = se::vec3(smat.data[0][0], smat.data[1][0], smat.data[2][0]).length();
  s->y = se::vec3(smat.data[0][1], smat.data[1][1], smat.data[2][1]).length();
  s->z = se::vec3(smat.data[0][2], smat.data[1][2], smat.data[2][2]).length();
}

auto decompose(se::mat4 const& m, se::vec3* t, se::vec3* r, se::vec3* s) noexcept -> void {
  // Extract translation T from transformation matrix
  // which could be found directly from matrix
  t->x = m.data[0][3];
  t->y = m.data[1][3];
  t->z = m.data[2][3];

  // Compute new transformation matrix M without translation
  se::mat4 M = m;
  for (int i = 0; i < 3; i++) M.data[i][3] = M.data[3][i] = 0.f;
  M.data[3][3] = 1.f;

  // Extract rotation R from transformation matrix
  // use polar decomposition, decompose into R&S by averaging M with its
  // inverse transpose until convergence to get R (because pure rotation
  // matrix has similar inverse and transpose)
  float norm;
  int count = 0;
  se::mat4 R = M;
  do {
    // Compute next matrix Rnext in series
    se::mat4 rNext;
    se::mat4 rInvTrans = inverse(transpose(R));
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

  se::vec3 euler = rotationMatrixToEulerAngles(se::mat3{R});
  r->x = euler.x * 180. / se::double_Pi;
  r->y = euler.y * 180. / se::double_Pi;
  r->z = euler.z * 180. / se::double_Pi;

  // Compute scale S using rotationand original matrix
  se::mat4 invR = inverse(R);
  se::mat4 Mtmp = M;
  se::mat4 smat = mul(invR, Mtmp);
  s->x = se::vec3(smat.data[0][0], smat.data[1][0], smat.data[2][0]).length();
  s->y = se::vec3(smat.data[0][1], smat.data[1][1], smat.data[2][1]).length();
  s->z = se::vec3(smat.data[0][2], smat.data[1][2], smat.data[2][2]).length();
}

auto AnimationCurve::evaluate(float time) noexcept -> float {
  if (time > keyFrames.back().time) {  // right warp
    switch (preWrapMode) {
      case WrapMode::CLAMP:
        return keyFrames.back().value;
        break;
      case WrapMode::REPEAT: {
        int passCount = int((time - keyFrames.back().time) /
                            (keyFrames.back().time - keyFrames.front().time));
        time = time - (passCount + 1) *
                          (keyFrames.back().time - keyFrames.front().time);
      } break;
      case WrapMode::PINGPOMG: {
        int passCount = int((time - keyFrames.back().time) /
                            (keyFrames.back().time - keyFrames.front().time));
        bool needReverse = (passCount % 2 == 0);
        time = time - (passCount + 1) *
                          (keyFrames.back().time - keyFrames.front().time);
        if (needReverse)
          time = keyFrames.front().time + keyFrames.back().time - time;
      } break;
      default:
        break;
    }
  } else if (time < keyFrames.front().time) {  // left warp
    switch (preWrapMode) {
      case WrapMode::CLAMP:
        return keyFrames.front().value;
        break;
      case WrapMode::REPEAT: {
        int passCount = int((keyFrames.front().time - time) /
                            (keyFrames.back().time - keyFrames.front().time));
        time = time + (passCount + 1) *
                          (keyFrames.back().time - keyFrames.front().time);
      } break;
      case WrapMode::PINGPOMG: {
        int passCount = int((keyFrames.front().time - time) /
                            (keyFrames.back().time - keyFrames.front().time));
        bool needReverse = (passCount % 2 == 0);
        time = time + (passCount + 1) *
                          (keyFrames.back().time - keyFrames.front().time);
        if (needReverse)
          time = keyFrames.front().time + keyFrames.back().time - time;
      } break;
      default:
        break;
    }
  }

  int left = 0;
  while (left + 1 < keyFrames.size()) {
    if (keyFrames[left].time <= time && keyFrames[left + 1].time > time) break;
    left++;
  }

  float t_l = 0;
  float t_r = 1;
  while (true) {
    float t = 0.5f * (t_l + t_r);
    Point point = evaluate(keyFrames[left], keyFrames[left + 1], t);
    float error = std::abs(point.time - time);
    if (error < errorTolerence)
      return point.value;
    else if (point.time < time)
      t_l = t;
    else
      t_r = t;
  }
}

auto AnimationCurve::evaluate(KeyFrame const& keyframe0,
                              KeyFrame const& keyframe1, float t) noexcept
    -> Point {
  // regular Cubic Hermite spline with tangents defined by hand
  float dt = keyframe1.time - keyframe0.time;
  float m0 = keyframe0.outTangent * dt;
  float m1 = keyframe1.inTangent * dt;

  float t2 = t * t;
  float t3 = t2 * t;

  float a = 2 * t3 - 3 * t2 + 1;
  float b = t3 - 2 * t2 + t;
  float c = t3 - t2;
  float d = -2 * t3 + 3 * t2;

  float time = a * keyframe0.time + b * m0 + c * m1 + d * keyframe1.time;
  float value = a * keyframe0.value + b * m0 + c * m1 + d * keyframe1.value;
  return Point{time, value};
}

auto compareKeyFrameByTime(KeyFrame const& lhv, KeyFrame const& rhv) noexcept
    -> bool {
  return lhv.time < rhv.time;
}

auto AnimationCurve::sortAllKeyFrames() noexcept -> void {
  std::sort(keyFrames.begin(), keyFrames.end(), compareKeyFrameByTime);
}
}