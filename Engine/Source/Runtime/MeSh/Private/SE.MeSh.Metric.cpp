#include "../Public/SE.MeSh.Metric.hpp"
#include <SE.Math.Misc.hpp>

namespace SIByL::MeSh {

void Quadric3::init(double a, double b, double c, double d, double area) {
  a2 = a * a;
  ab = a * b;
  ac = a * c;
  ad = a * d;
  b2 = b * b;
  bc = b * c;
  bd = b * d;
  c2 = c * c;
  cd = c * d;
  d2 = d * d;
  r = area;
}

void Quadric3::init(Math::dmat4 const& Q, double area) {
  a2 = Q.data[0][0];
  ab = Q.data[0][1];
  ac = Q.data[0][2];
  ad = Q.data[0][3];
  b2 = Q.data[1][1];
  bc = Q.data[1][2];
  bd = Q.data[1][3];
  c2 = Q.data[2][2];
  cd = Q.data[2][3];
  d2 = Q.data[3][3];
  r = area;
}

Math::dmat3 Quadric3::tensor() const {
  return Math::dmat3(Math::dvec3(a2, ab, ac), Math::dvec3(ab, b2, bc),
                     Math::dvec3(ac, bc, c2));
}

Math::dmat4 Quadric3::homogeneous() const {
  return Math::dmat4(Math::dvec4(a2, ab, ac, ad), Math::dvec4(ab, b2, bc, bd),
                     Math::dvec4(ac, bc, c2, cd), Math::dvec4(ad, bd, cd, d2));
}

void Quadric3::set_coefficients(const double* v) {
  a2 = v[0];
  ab = v[1];
  ac = v[2];
  ad = v[3];
  b2 = v[4];
  bc = v[5];
  bd = v[6];
  c2 = v[7];
  cd = v[8];
  d2 = v[9];
}

void Quadric3::point_constraint(const float* p) {
  // A point constraint quadric measures the squared distance
  // of any point v to the given point p.

  a2 = b2 = c2 = 1.0;
  ab = ac = bc = 0.0;  // A = I
  ad = -p[0];
  bd = -p[1];
  cd = -p[2];                                    // b = -p
  d2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];  // c = p*p
}

Quadric3& Quadric3::operator=(Quadric3 const& Q) {
  r = Q.r;
  a2 = Q.a2;
  ab = Q.ab;
  ac = Q.ac;
  ad = Q.ad;
  b2 = Q.b2;
  bc = Q.bc;
  bd = Q.bd;
  c2 = Q.c2;
  cd = Q.cd;
  d2 = Q.d2;
  return *this;
}

Quadric3& Quadric3::operator+=(Quadric3 const& Q) {
  // Accumulate area
  r += Q.r;
  // Accumulate coefficients
  a2 += Q.a2;
  ab += Q.ab;
  ac += Q.ac;
  ad += Q.ad;
  b2 += Q.b2;
  bc += Q.bc;
  bd += Q.bd;
  c2 += Q.c2;
  cd += Q.cd;
  d2 += Q.d2;
  return *this;
}

Quadric3& Quadric3::operator-=(Quadric3 const& Q) {
  r -= Q.r;
  a2 -= Q.a2;
  ab -= Q.ab;
  ac -= Q.ac;
  ad -= Q.ad;
  b2 -= Q.b2;
  bc -= Q.bc;
  bd -= Q.bd;
  c2 -= Q.c2;
  cd -= Q.cd;
  d2 -= Q.d2;
  return *this;
}

Quadric3& Quadric3::operator*=(double s) {
  // Scale coefficients
  a2 *= s;
  ab *= s;
  ac *= s;
  ad *= s;
  b2 *= s;
  bc *= s;
  bd *= s;
  c2 *= s;
  cd *= s;
  d2 *= s;
  return *this;
}

Quadric3& Quadric3::transform(const Math::dmat4& P) {
  Math::dmat4 Q = homogeneous();
  Math::dmat4 Pa = adjoint(P);
  // Compute:  trans(Pa) * Q * Pa
  // NOTE: Pa is symmetric since Q is symmetric
  Q = Pa * Q * Pa;
  // ??BUG: Should we be transforming the area??
  init(Q, r);
  return *this;
}

double Quadric3::evaluate(double x, double y, double z) const {
  // Evaluate vAv + 2bv + c
  return x * x * a2 + 2 * x * y * ab + 2 * x * z * ac + 2 * x * ad +
         y * y * b2 + 2 * y * z * bc + 2 * y * bd + z * z * c2 + 2 * z * cd +
         d2;
}

bool Quadric3::optimize(Math::dvec3& v) const {
  Math::dmat3 Ainv;
  double det = invert(Ainv, tensor());
  if (Math::float_equal(det, 0.0, 1e-12)) return false;

  v = -(Ainv * vector());
  return true;
}

bool Quadric3::optimize(float* x, float* y, float* z) const {
  Math::dvec3 v;

  bool success = optimize(v);
  if (success) {
    *x = (float)v[0];
    *y = (float)v[1];
    *z = (float)v[2];
  }
  return success;
}

bool Quadric3::optimize(Math::dvec3& v, const Math::dvec3& v1,
                        const Math::dvec3& v2) const {
  Math::dvec3 d = v1 - v2;
  Math::dmat3 A = tensor();

  Math::dvec3 Av2 = A * v2;
  Math::dvec3 Ad = A * d;

  double denom = 2.0 * Math::dot(d, Ad);
  if (Math::float_equal(denom, 0.0, 1e-12)) return false;

  double a =
      (-2 * Math::dot(vector(), d) - Math::dot(d, Av2) - Math::dot(v2, Ad)) /
      (2 * Math::dot(d, Ad));

  if (a < 0.0)
    a = 0.0;
  else if (a > 1.0)
    a = 1.0;

  v = a * d + v2;
  return true;
}

bool Quadric3::optimize(Math::dvec3& v, const Math::dvec3& v1,
                        const Math::dvec3& v2, const Math::dvec3& v3) const {
  Math::dvec3 d13 = v1 - v3;
  Math::dvec3 d23 = v2 - v3;
  Math::dmat3 A = tensor();
  Math::dvec3 B = vector();

  Math::dvec3 Ad13 = A * d13;
  Math::dvec3 Ad23 = A * d23;
  Math::dvec3 Av3 = A * v3;

  double d13_d23 = Math::dot(d13, Ad23) + Math::dot(d23, Ad13);
  double v3_d13 = Math::dot(d13, Av3) + Math::dot(v3, Ad13);
  double v3_d23 = Math::dot(d23, Av3) + Math::dot(v3, Ad23);

  double d23Ad23 = Math::dot(d23, Ad23);
  double d13Ad13 = Math::dot(d13, Ad13);

  double denom = d13Ad13 * d23Ad23 - 2 * d13_d23;
  if (Math::float_equal(denom, 0.0, 1e-12)) return false;

  double a = (d23Ad23 * (2 * Math::dot(B, d13) + v3_d13) -
              d13_d23 * (2 * Math::dot(B, d23) + v3_d23)) /
             -denom;

  double b = (d13Ad13 * (2 * Math::dot(B, d23) + v3_d23) -
              d13_d23 * (2 * Math::dot(B, d13) + v3_d13)) /
             -denom;

  if (a < 0.0)
    a = 0.0;
  else if (a > 1.0)
    a = 1.0;
  if (b < 0.0)
    b = 0.0;
  else if (b > 1.0)
    b = 1.0;

  v = a * d13 + b * d23 + v3;
  return true;
}
}