#pragma once
#include <SE.Math.Geometric.hpp>
#include "SE.MeSh.Metric.hpp"

namespace SIByL::MeSh {

struct Frame {
  Math::dvec3 v0;       // origin of local frame
  Math::dvec3 axes[3];  // unit axis vectors of local frame
  double d0;            // together with axis(2), defines base plane

  Frame();
  Frame(Math::dvec3 const &c);
  Frame(Math::dvec3 const &c, Math::dvec3 const &u0, Math::dvec3 const &u1,
        Math::dvec3 const &u2);
  auto origin() noexcept -> Math::dvec3& { return v0; }
  auto origin() const noexcept -> Math::dvec3 const& { return v0; }
  auto axis(uint32_t i) noexcept -> Math::dvec3&;
  auto axis(uint32_t i) const noexcept -> Math::dvec3 const&;
  auto plane_offset() const noexcept -> double { return d0; }
  auto plane_offset(double d) noexcept -> void { d0 = d; }

  auto to_frame(Math::dvec3 const &v, Math::dvec3 const &u) const noexcept
      -> Math::dvec3;
  auto from_frame(Math::dvec3 const &u, Math::dvec3 const &v) const noexcept
      -> Math::dvec3;
  auto compute_xform_toframe() const noexcept -> Math::dmat4;
  auto compute_xform_fromframe() const noexcept -> Math::dmat4;

  auto dist_to_plane(Math::dvec3 const &v) const noexcept -> double;
  auto align_axis(uint32_t i, Math::dvec3 const &v) noexcept -> void;
};

struct FitFrame : public Frame {
  Math::dvec3 vmin, vmax;	// Bounding box in local frame
  Math::dvec3 axis_evals;	// Eigenvalues from prin. component analysis
  Math::dvec3 normal_accum, avg_normal;

  FitFrame();
  FitFrame(Math::dvec3 const &c,  Math::dvec3 const &u0, 
           Math::dvec3 const &u1, Math::dvec3 const &u2);
  FitFrame(Quadric3 const &, uint32_t nverts);

  auto compute_frame(Quadric3 const &, uint32_t nverts) noexcept -> bool;
  auto compute_frame(const float *, const float *, const float *) noexcept
      -> bool;

  auto total_normal() noexcept -> Math::dvec3& { return normal_accum; }
  auto total_normal() const noexcept -> Math::dvec3 const& { return normal_accum; }

  void set_normal(Math::dvec3 const& n) noexcept;
  void add_normal(Math::dvec3 const& n) noexcept;
  void clear_normal() noexcept;
  void finalize_normal() noexcept;

  bool compute_frame(Quadric3 const &Q_fit, uint32_t _nverts);
  bool compute_frame(Math::dvec3 const &, Math::dvec3 const &,
                     Math::dvec3 const &);

  void reset_bounds();
  void accumulate_bounds(const FitFrame &);
  void accumulate_bounds(const float *, uint32_t npoint = 1);
  void accumulate_bounds(const double *, uint32_t npoint = 1);
};


}