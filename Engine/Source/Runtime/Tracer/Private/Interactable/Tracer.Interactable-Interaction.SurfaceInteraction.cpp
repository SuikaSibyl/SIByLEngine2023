module;
#include <cmath>
module Tracer.Interactable:Interaction.SurfaceInteraction;
import Tracer.Interactable;
import Core.Memory;
import Math.Vector;
import Math.Geometry;
import Math.Transform;
import Math.EquationSolving;
import Tracer.Shape;
import Tracer.Ray;

namespace SIByL::Tracer
{
	SurfaceInteraction::SurfaceInteraction()
		: Interaction(Math::vec3{}, Math::normal3{}, Math::vec3{}, Math::vec3{}, 0, nullptr)
	{}

	SurfaceInteraction::SurfaceInteraction(Math::point3 const& p, Math::point3 const& pError,
		Math::point2 const& uv, Math::vec3 const& wo,
		Math::vec3 const& dpdu, Math::vec3 const& dpdv,
		Math::normal3 const& dndu, Math::vec3 const& dndv,
		float time, Shape const* shape)
		: Interaction(p, Math::normal3(Math::normalize(Math::cross(dpdu, dpdv))), pError, wo, time, nullptr)
		, uv(uv), dpdu(dpdu), dpdv(dpdv), dndu(dndu), dndv(dndv), shape(shape)
	{
		// Initialize shading geometry from true geometry
		shading.n = n;
		shading.dpdu = dpdu;
		shading.dpdv = dpdv;
		shading.dndu = dndu;
		shading.dndv = dndv;
		// Adjust normal based on orientationand handedness
		if (shape && (shape->reverseOrientation ^ shape->transformSwapsHandedness)) {
			n *= -1;
			shading.n *= -1;
		}
	}

	auto SurfaceInteraction::setShadingGeometry(Math::vec3 const& dpdus, Math::vec3 const& dpdvs,
		Math::normal3 const& dndus, Math::normal3 const& dndvs, bool orientationIsAuthoritative) noexcept -> void {
		// Compute shading.n for SurfaceInteraction
		shading.n = Math::normalize((Math::normal3)Math::cross(dpdus, dpdvs));
		if (shape && (shape->reverseOrientation ^ shape->transformSwapsHandedness))
			shading.n = -shading.n;
		if (orientationIsAuthoritative)
			n = Math::faceforward(n, shading.n);
		else
			shading.n = Math::faceforward(shading.n, n);
		// Initialize shading partial derivative values
		shading.dpdu = dpdus;
		shading.dpdv = dpdvs;
		shading.dndu = dndus;
		shading.dndv = dndvs;
	}

	auto SurfaceInteraction::computeScatteringFunctions(RayDifferential const& ray, Core::MemoryArena& arena, bool allowMultipleLobes, TransportMode mode) noexcept -> void {
		computeDifferentials(ray);
		primitive->computeScatteringFunctions(this, arena, mode, allowMultipleLobes);
	}

	auto SurfaceInteraction::computeDifferentials(RayDifferential const& ray) const noexcept -> void {
		if (ray.hasDifferentials) {
			// estimate screen space change in p and (u.v)
			// compute auxiliary intersection points with plane
			float d = Math::dot(n, Math::vec3(p.x, p.y, p.z));
			float tx = -(Math::dot(n, Math::vec3(ray.rxOrigin)) - d) / Math::dot(n, ray.rxDirection);
			Math::point3 px = ray.rxOrigin + tx * ray.rxDirection;
			float ty = -(Math::dot(n, Math::vec3(ray.ryOrigin)) - d) / Math::dot(n, ray.ryDirection);
			Math::point3 py = ray.ryOrigin + ty * ray.ryDirection;
			dpdx = px - p;
			dpdy = py - p;
			// compute (u,v) offsets at auxiliary points
			// choose two dimensions to use for ray offset computation
			int dim[2];
			if (std::abs(n.x) > std::abs(n.y) && std::abs(n.x) > std::abs(n.z)) {
				dim[0] = 1; dim[1] = 2;
			}
			else if (std::abs(n.y) > std::abs(n.z)) {
				dim[0] = 0; dim[1] = 2;
			}
			else {
				dim[0] = 0; dim[1] = 1;
			}
			// initialize A,Bx, and By matrices for offset computation
			float A[2][2] = {
				{dpdu[dim[0]],dpdv[dim[0]]},
			};
			float Bx[2] = { px[dim[0]] - p[dim[0]],px[dim[1]] - p[dim[1]] };
			float By[2] = { py[dim[0]] - p[dim[0]],py[dim[1]] - p[dim[1]] };
			if (!Math::solveLinearSystem2x2(A, Bx, &dudx, &dvdx))
				dudx = dvdx = 0;
			if (!Math::solveLinearSystem2x2(A, By, &dudy, &dvdy))
				dudy = dvdy = 0;
		}
		else {
			dudx = dvdx = 0;
			dudy = dvdy = 0;
			dpdx = dpdy = Math::vec3(0, 0, 0);
		}
	}

	auto SurfaceInteraction::Le(Math::vec3 const& w) const noexcept -> Spectrum {
		//AreaLight const* area = primitive->getAreaLight();
		//return area ? area->L(*this, w) : Spectrum(0.f);
	}
}


namespace SIByL::Math
{
	inline auto operator*(Transform const& t, Tracer::SurfaceInteraction const& si)->Tracer::SurfaceInteraction {
		Tracer::SurfaceInteraction r = si;
		// Transform pand pError in SurfaceInteraction
		r.p = t(si.p, si.pError, r.pError);
		// Transform remaining members of SurfaceInteraction
		r.n = Math::normalize(t * si.n);
		r.wo = Math::normalize(t * si.wo);
		r.dpdu = t * si.dpdu;
		r.dpdv = t * si.dpdv;
		r.dndu = t * si.dndu;
		r.dndv = t * si.dndv;
		r.shading.n = Math::normalize(t * si.shading.n);
		r.shading.dpdu = t * si.shading.dpdu;
		r.shading.dpdv = t * si.shading.dpdv;
		r.shading.dndu = t * si.shading.dndu;
		r.shading.dndv = t * si.shading.dndv;
		r.dpdx = t * si.dpdx;
		r.dpdy = t * si.dpdy;
		r.shading.n = faceforward(r.shading.n, r.n);
		return r;
	}

	inline auto operator*(AnimatedTransform const& t, Tracer::SurfaceInteraction const& si)->Tracer::SurfaceInteraction {
		return si;
	}
}
