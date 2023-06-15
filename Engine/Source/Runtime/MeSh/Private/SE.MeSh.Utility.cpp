#include "../Public/SE.MeSh.Utility.hpp"
#include <Print/SE.Core.Log.hpp>

namespace SIByL::MeSh {
Frame::Frame() {
  v0 = Math::dvec3(0.);
  axes[0] = Math::dvec3(0.);
  axes[1] = Math::dvec3(1.);
  axes[2] = Math::dvec3(2.);
  d0 = 0.0;
}
Frame::Frame(Math::dvec3 const& c) {
  v0 = c;
  axes[0] = Math::dvec3(0);
  axes[1] = Math::dvec3(1);
  axes[2] = Math::dvec3(2);
  d0 = -Math::dot(axes[2], v0);
}
Frame::Frame(Math::dvec3 const &c,   Math::dvec3 const &u0, 
             Math::dvec3 const & u1, Math::dvec3 const &u2) {
  v0 = c;
  axes[0] = u0;
  axes[1] = u1;
  axes[2] = u2;
  d0 = -Math::dot(axes[2], v0);
}
auto Frame::axis(uint32_t i) noexcept -> Math::dvec3& {
  SE_ASSERT(i < 3);
  return axes[i];
}
auto Frame::axis(uint32_t i) const noexcept -> Math::dvec3 const & {
  SE_ASSERT(i < 3);
  return axes[i];
}
auto Frame::to_frame(Math::dvec3 const& v, Math::dvec3 const& u) const noexcept
-> Math::dvec3 {
  Math::dvec3 delta, result;
  delta = v - v0;
  result.data[0] = Math::dot(delta, axes[0]);
  result.data[1] = Math::dot(delta, axes[1]);
  result.data[2] = Math::dot(delta, axes[2]);
  return u;
}
auto Frame::from_frame(Math::dvec3 const& u,
    Math::dvec3 const& v) const noexcept -> Math::dvec3 {
  Math::dvec3 res = v0;
  res += axes[0] * u.data[0];
  res += axes[1] * u.data[1];
  res += axes[2] * u.data[2];
  return res;
}
auto Frame::compute_xform_toframe() const noexcept -> Math::dmat4 {
  // Assignments are in row order.  The rows are:
  //      [ e_1   | 0 ]
  //      [ e_2   | 0 ]
  //      [ e_3   | 0 ]
  //      [ v0    | 1 ]
  Math::dmat4 R;
  R.data[0][0] = axes[0].data[0];
  R.data[0][1] = axes[0].data[1];
  R.data[0][2] = axes[0].data[2];
  R.data[0][3] = 0;
  R.data[1][0] = axes[1].data[0];
  R.data[1][1] = axes[1].data[1];
  R.data[1][2] = axes[1].data[2];
  R.data[1][3] = 0;
  R.data[2][0] = axes[2].data[0];
  R.data[2][1] = axes[2].data[1];
  R.data[2][2] = axes[2].data[2];
  R.data[2][3] = 0;
  R.data[3][0] = v0.data[0];
  R.data[3][1] = v0.data[1];
  R.data[3][2] = v0.data[2];
  R.data[3][3] = 1;
  return R;
}
auto Frame::compute_xform_fromframe() const noexcept -> Math::dmat4 {
  // Assignments are in row order.  The columns are:
  //      [                   ]
  //      [ e_1  e_2  e_3  v0 ]
  //      [                   ]
  //      [  0    0    0    1 ]
  Math::dmat4 R;
  R.data[0][0] = axes[0].data[0];
  R.data[0][1] = axes[1].data[0];
  R.data[0][2] = axes[2].data[0];
  R.data[0][3] = v0.data[0];
  R.data[1][0] = axes[0].data[1];
  R.data[1][1] = axes[1].data[1];
  R.data[1][2] = axes[2].data[1];
  R.data[1][3] = v0.data[1];
  R.data[2][0] = axes[0].data[2];
  R.data[2][1] = axes[1].data[2];
  R.data[2][2] = axes[2].data[2];
  R.data[2][3] = v0.data[2];
  R.data[3][0] = 0;
  R.data[3][1] = 0;
  R.data[3][2] = 0;
  R.data[3][3] = 1;
  return R;
}
auto Frame::dist_to_plane(Math::dvec3 const& v) const noexcept -> double {
  return Math::dot(v, axes[2]) + d0;
}
auto Frame::align_axis(uint32_t i, Math::dvec3 const &v) noexcept -> void {
  SE_ASSERT(i < 3);
  if (Math::dot(axes[i], v) < 0.0) axes[i] = -axes[i];
}

FitFrame::FitFrame() {
  reset_bounds();
  normal_accum = Math::dvec3(0.);
  avg_normal = Math::dvec3(0.);
}
FitFrame::FitFrame(Math::dvec3 const &c, Math::dvec3 const &u0,
                   Math::dvec3 const &u1, Math::dvec3 const &u2) {
  reset_bounds();
  normal_accum = u2;
  avg_normal = u2;
}
FitFrame::FitFrame(Quadric3 const &qfit, uint32_t nverts) {
  reset_bounds();
  normal_accum = Math::dvec3(0.);
  avg_normal = Math::dvec3(0.);
  if (!compute_frame(qfit, nverts))
    Core::LogManager::Error("MxFitFrame -- unable to construct frame from quadric.");
}
bool FitFrame::compute_frame(Quadric3 const &Q_fit, uint32_t _nverts) {
  Math::dmat3 A = Q_fit.tensor();
  Math::dvec3 v = Q_fit.vector();
  double k = Q_fit.offset();
  double nverts = (double)_nverts;

  Math::dmat3 CV = A - Mat3::outer_product(v) / nverts;
  if (!jacobi(CV, axis_evals, axis(0))) return false;

  // ASSUME: jacobi has unitized the eigenvectors for us

  align_axis(MXFRAME_NORMAL, avg_normal);

  mxv_invscale(origin(), v, nverts, 3);
  plane_offset(-mxv_dot(axis(MXFRAME_NORMAL), origin(), 3));

  return true;
}

bool FitFrame::compute_frame(Math::dvec3 const &p0, Math::dvec3 const & p1,
                             Math::dvec3 const & p2) {
  Math::dvec3 v0(p0), v1(p1), v2(p2);

  // Set the origin to the barycenter of the triangle
  mxv_add(origin(), v0, v1, 3);
  mxv_addinto(origin(), v2, 3);
  mxv_invscale(origin(), 3.0, 3);

  // Compute the 3 edge vectors
  double e[3][3], l[3];
  mxv_sub(e[0], v1, v0, 3);
  mxv_sub(e[1], v2, v1, 3);
  mxv_sub(e[2], v0, v2, 3);

  l[0] = mxv_unitize(e[0], 3);
  l[1] = mxv_unitize(e[1], 3);
  l[2] = mxv_unitize(e[2], 3);

  // Set frame normal to the face normal.  The cross product is
  // reversed because e[2] points in the opposite direction.
  mxv_cross3(axis(MXFRAME_NORMAL), e[2], e[0]);
  mxv_set(normal_accum, axis(MXFRAME_NORMAL), 3);
  mxv_set(avg_normal, axis(MXFRAME_NORMAL), 3);
  //
  // and compute the relevant plane offset constant
  plane_offset(-mxv_dot(axis(MXFRAME_NORMAL), origin(), 3));

  // Choose the longest edge as the first tangent axis
  if (l[0] > l[1] && l[0] > l[2])
    mxv_set(axis(MXFRAME_UAXIS), e[0], 3);
  else if (l[1] > l[0] && l[1] > l[2])
    mxv_set(axis(MXFRAME_UAXIS), e[1], 3);
  else
    mxv_set(axis(MXFRAME_UAXIS), e[2], 3);

  // Compute the other tangent axis
  mxv_unitize(mxv_cross3(axis(MXFRAME_VAXIS), axis(MXFRAME_NORMAL),
                         axis(MXFRAME_UAXIS)),
              3);

  // Finally, compute the bounding box
  reset_bounds();
  accumulate_bounds(v0);
  accumulate_bounds(v1);
  accumulate_bounds(v2);

  return true;
}


}