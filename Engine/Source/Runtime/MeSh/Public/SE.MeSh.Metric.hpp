#pragma once
#include <SE.Math.Geometric.hpp>

namespace SIByL::MeSh {
struct Quadric3 {
  double a2, ab, ac, ad;
  double b2, bc, bd;
  double c2, cd;
  double d2;

  double r;

  void init(double a, double b, double c, double d, double area);
  void init(Math::dmat4 const& Q, double area);

  Quadric3() { clear(); }
  Quadric3(double a, double b, double c, double d, double area = 1.0) {
    init(a, b, c, d, area);
  }
  Quadric3(const float* n, double d, double area = 1.0) {
    init(n[0], n[1], n[2], d, area);
  }
  Quadric3(const double* n, double d, double area = 1.0) {
    init(n[0], n[1], n[2], d, area);
  }
  Quadric3(const Quadric3& Q) { *this = Q; }

  Math::dmat3 tensor() const;
  Math::dvec3 vector() const { return Math::dvec3(ad, bd, cd); }
  double offset() const { return d2; }
  double area() const { return r; }
  Math::dmat4 homogeneous() const;

  void set_coefficients(const double*);
  void set_area(double a) { r = a; }
  void point_constraint(const float*);

  void clear(double val = 0.0) {
    a2 = ab = ac = ad = b2 = bc = bd = c2 = cd = d2 = r = val;
  }
  Quadric3& operator=(const Quadric3& Q);
  Quadric3& operator+=(const Quadric3& Q);
  Quadric3& operator-=(const Quadric3& Q);
  Quadric3& operator*=(double s);
  Quadric3& transform(const Math::dmat4& P);

  double evaluate(double x, double y, double z) const;
  double evaluate(const double* v) const { return evaluate(v[0], v[1], v[2]); }
  double evaluate(const float* v) const { return evaluate(v[0], v[1], v[2]); }

  double operator()(double x, double y, double z) const {
    return evaluate(x, y, z);
  }
  double operator()(const double* v) const {
    return evaluate(v[0], v[1], v[2]);
  }
  double operator()(const float* v) const { return evaluate(v[0], v[1], v[2]); }

  bool optimize(Math::dvec3& v) const;
  bool optimize(float* x, float* y, float* z) const;
  bool optimize(Math::dvec3& v, Math::dvec3 const& v1, Math::dvec3 const& v2) const;
  bool optimize(Math::dvec3& v, Math::dvec3 const& v1, Math::dvec3 const& v2,
                Math::dvec3 const& v3) const;
};
}