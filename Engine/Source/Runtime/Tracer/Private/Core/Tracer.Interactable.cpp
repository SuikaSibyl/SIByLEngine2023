module;
#include <cmath>
#include <cstdint>
#include <algorithm>
module Tracer.Interactable;
import SE.Core.Memory;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Ray;
import Tracer.BxDF;
import Tracer.Medium;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	// ****************************
	// | Interaction Categories	  |
	// ****************************
	auto Interaction::isSurfaceInteraction() const noexcept -> bool {
		return n != Math::normal3();
	}

	auto Interaction::isMediumInteraction() const noexcept -> bool {
		return !isSurfaceInteraction();
	}

	auto Interaction::getMedium(Math::vec3 const& w) const noexcept -> Medium const* {
		return Math::dot(w, n) > 0 ? mediumInterface.outside : mediumInterface.inside;
	}

	auto Interaction::spawnRay(Math::vec3 const& d) const noexcept -> Ray {
		Math::point3 o = offsetRayOrigin(p, pError, n, d);
		return Ray(o, d, Math::float_infinity, time, getMedium(d));
	}

	auto Interaction::spawnRayTo(Interaction const& it) const noexcept -> Ray {
		Math::point3 origin = offsetRayOrigin(p, pError, n, it.p - p);
		Math::point3 target = offsetRayOrigin(it.p, it.pError, it.n, origin - it.p);
		Math::vec3 d = target - origin;
		return Ray(origin, d, 1 - Math::shadow_epsilon, time, getMedium(d));
	}

	auto Interaction::spawnRayTo(Math::point3 const& p2) const noexcept -> Ray {
		float const shadowEpsilon = 0.0001f;
		Math::point3 origin = offsetRayOrigin(p, pError, n, p2 - p);
		Math::vec3 d = p2 - origin;
		return Ray(origin, d, 1 - shadowEpsilon, time, getMedium(d));
	}

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
		// check hasDifferentials field to avoid some rays (e.g. rays starting from light sourcces)
		if (ray.hasDifferentials) {
			// estimate screen space change in p and (u.v)
			//  compute auxiliary intersection points with plane
			float d = Math::dot(n, Math::vec3(p.x, p.y, p.z));
			float tx = -(Math::dot(n, Math::vec3(ray.rxOrigin)) - d) / Math::dot(n, ray.rxDirection);
			Math::point3 px = ray.rxOrigin + tx * ray.rxDirection;
			float ty = -(Math::dot(n, Math::vec3(ray.ryOrigin)) - d) / Math::dot(n, ray.ryDirection);
			Math::point3 py = ray.ryOrigin + ty * ray.ryDirection;
			dpdx = px - p;
			dpdy = py - p;
			//  compute (u,v) offsets at auxiliary points
			// ─ choose two dimensions to use for ray offset computation
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
			// ─ initialize A,Bx, and By matrices for offset computation
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
			// if the differentials are not present, then the derivatives are all set to zero, 
			// which will eventually lead to unfiltered point sampling of textures.
			dudx = dvdx = 0;
			dudy = dvdy = 0;
			dpdx = dpdy = Math::vec3(0, 0, 0);
		}
	}

	auto SurfaceInteraction::Le(Math::vec3 const& w) const noexcept -> Spectrum {
		AreaLight const* area = primitive->getAreaLight();
		return area ? area->L(*this, w) : Spectrum(0.f);
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

namespace SIByL::Tracer
{
	// ****************************
	// | Material Categories	  |
	// ****************************

	BSDF::BSDF(SurfaceInteraction const& si, float eta)
		:eta(eta), ns(si.shading.n), ng(si.n)
		, ss(Math::normalize(si.shading.dpdu)), ts(Math::cross(ns, ss)) {}

	auto BSDF::add(BxDF* b) noexcept -> void {
		bxdfs[nBxDFs++] = b;
	}

	auto BSDF::numComponents(BxDF::Type flags) const noexcept -> int {
		int num = 0;
		for (int i = 0; i < nBxDFs; ++i)
			if (bxdfs[i]->matchFlags(flags)) ++num;
		return num;
	}

	auto BSDF::worldToLocal(Math::vec3 const& v) const noexcept -> Math::vec3 {
		return Math::vec3(Math::dot(v, ss), Math::dot(v, ts), Math::dot(v, ns));
	}

	auto BSDF::localToWorld(Math::vec3 const& v) const noexcept -> Math::vec3 {
		return Math::vec3(ss.x * v.x + ts.x * v.y + ns.x * v.z,
			ss.y * v.x + ts.y * v.y + ns.y * v.z,
			ss.z * v.x + ts.z * v.y + ns.z * v.z);
	}

	auto BSDF::f(Math::vec3 const& woW, Math::vec3 const& wiW, BxDF::Type flags) const noexcept -> Spectrum {
		Math::vec3 wi = worldToLocal(wiW), wo = worldToLocal(woW);
		bool reflect = Math::dot(wiW, ng) * Math::dot(woW, ng) > 0;
		Spectrum f(0.f);
		for (int i = 0; i < nBxDFs; ++i)
			if (bxdfs[i]->matchFlags(flags) &&
				((reflect && (bxdfs[i]->type & BxDF::BSDF_REFLECTION)) ||
					(!reflect && (bxdfs[i]->type & BxDF::BSDF_TRANSMISSION))))
				f += bxdfs[i]->f(wo, wi);
		return f;
	}

	auto BSDF::sample_f(Math::vec3 const& woWorld, Math::vec3* wiWorld, Math::point2 const& u, float* pdf, BxDF::Type type, BxDF::Type* sampledType) const noexcept -> Spectrum {
		// Choose which BxDF to sample
		int const matchingComps = numComponents(type);
		if (matchingComps == 0) {
			*pdf = 0;
			return Spectrum(0);
		}
		//  uses the first dimension of the provided u sample to select one of the components with equal probability
		int comp = std::min((int)std::floor(u[0] * matchingComps), matchingComps - 1);
		//  get BxDF pointer for chosen component
		BxDF* bxdf = nullptr;
		int count = comp;
		for (int i = 0; i < nBxDFs; ++i)
			if (bxdfs[i]->matchFlags(type) && count-- == 0) {
				bxdf = bxdfs[i];
				break;
			}
		// Remap BxDF sample u to[0, 1)^2, recover it to a uniform random value
		Math::point2 uRemapped(u[0] * matchingComps - comp, u[1]);
		// Sample chosen BxDF
		Math::vec3 wi, wo = worldToLocal(woWorld);
		*pdf = 0;
		if (sampledType) *sampledType = bxdf->type;
		Spectrum f = bxdf->sample_f(wo, &wi, uRemapped, pdf, sampledType);
		if (*pdf == 0)
			return 0;
		*wiWorld = localToWorld(wi);
		// Compute overall PDF with all matching BxDFs
		if (!(bxdf->type & BxDF::BSDF_SPECULAR) && matchingComps > 1) // need to skip perfectly specular cases
			for (int i = 0; i < nBxDFs; ++i)
				if (bxdfs[i] != bxdf && bxdfs[i]->matchFlags(type))
					// add in the contribution of the others
					*pdf += bxdfs[i]->pdf(wo, wi);
		if (matchingComps > 1) *pdf /= matchingComps;
		// Compute value of BSDF for sampled direction
		if (!(bxdf->type & BxDF::BSDF_SPECULAR) && matchingComps > 1) { // specular BxDF::f() would return 0, so skip
			bool const reflect = Math::dot(*wiWorld, ng) * Math::dot(woWorld, ng) > 0;
			f = 0.;
			for (int i = 0; i < nBxDFs; ++i)
				if (bxdfs[i]->matchFlags(type) &&
					// // only consider reflect, when on the same semishpere
					((reflect && (bxdfs[i]->type & BxDF::BSDF_REFLECTION)) ||
						// only consider transimission, when on different semishpere
						(!reflect && (bxdfs[i]->type & BxDF::BSDF_TRANSMISSION))))
					f += bxdfs[i]->f(wo, wi);
		}
		return f;
	}

	auto BSDF::pdf(Math::vec3 const& woWorld, Math::vec3 const& wiWorld, BxDF::Type flags) const noexcept -> float {
		if (nBxDFs == 0) return 0.f;
		Math::vec3 const wi = worldToLocal(wiWorld), wo = worldToLocal(woWorld);
		if (wo.z == 0) return 0.;
		float pdf = 0.f;
		int matchingComps = 0;
		for (int i = 0; i < nBxDFs; ++i)
			if (bxdfs[i]->matchFlags(flags)) {
				++matchingComps;
				pdf += bxdfs[i]->pdf(wo, wi);
			}
		return matchingComps > 0 ? pdf / matchingComps : 0.f;
	}

	// ****************************
	// | Light Categories	      |
	// ****************************
	auto VisibilityTester::unoccluded(Scene const& scene) const noexcept -> bool {
		return !scene.intersectP(_p0.spawnRayTo(_p1));
	}

	auto VisibilityTester::Tr(Scene const& scene, Sampler& sampler) const noexcept -> Spectrum {
		Ray ray(_p0.spawnRayTo(_p1));
		Spectrum Tr(1.f);
		while (true) {
			SurfaceInteraction isect;
			bool hitSurface = scene.intersect(ray, &isect);
			// Handle opaque surface along ray’s path
			if (hitSurface && isect.primitive->getMaterial() != nullptr)
				return Spectrum(0.0f);
			// Update transmittance for current ray segment
			if (ray.medium)
				Tr *= ray.medium->Tr(ray, sampler);
			// Generate next ray segment or return final transmittance
			if (!hitSurface)
				break;
			ray = isect.spawnRayTo(_p1);
		}
		return Tr;
	}

	Light::Light(int flags, Math::Transform const& LightToWorld,
		MediumInterface const& mediumInterface, int nSamples)
		: flags(flags), nSamples(std::max(1, nSamples))
		, mediumInterface(mediumInterface), lightToWorld(LightToWorld), worldToLight(Math::inverse(LightToWorld))
	{
		// Warn if light has transformation with non-uniform scale
	}

	// ****************************
	// |  Geometries Categories	  |
	// ****************************

	auto Shape::worldBound() const noexcept -> Math::bounds3 {
		return (*objectToWorld) * (objectBound());
	}

	auto Shape::intersectP(Ray const& ray, bool testAlphaTexture) const  -> bool {
		float tHit = ray.tMax;
		SurfaceInteraction isect;
		return intersect(ray, &tHit, &isect, testAlphaTexture);
	}

	auto Shape::pdf(Interaction const&) const noexcept -> float {
		return 1 / area();
	}

	auto Shape::sample(const Interaction& ref, Math::point2 const& u) const noexcept -> Interaction {
		return sample(u);
	}

	auto Shape::pdf(const Interaction& ref, Math::vec3 const& wi) const noexcept -> float {
		// Intersect sample ray with area light geometry
		Ray ray = ref.spawnRay(wi);
		float tHit;
		SurfaceInteraction isectLight;
		if (!intersect(ray, &tHit, &isectLight, false)) return 0;
		// Convert light sample weight to solid angle measure
		float pdf = Math::distanceSquared(ref.p, isectLight.p) / (absDot(isectLight.n, -wi) * area());
		return pdf;
	}
	
	// ****************************
	// |		  Scene           |
	// ****************************
	Scene::Scene(Primitive* aggregate, std::vector<Light*> const& lights)
		:lights(lights), aggregate(aggregate)
	{
		worldBound = aggregate->worldBound();
		for (auto const& light : lights)
			light->preprocess(*this);
	}

	auto Scene::getWorldBound() const noexcept -> Math::bounds3 const& {
		return worldBound;
	}

	auto Scene::intersect(Ray const& ray, SurfaceInteraction* isect) const noexcept -> bool {
		return aggregate->intersect(ray, isect);
	}

	auto Scene::intersectP(Ray const& ray) const noexcept -> bool {
		return aggregate->intersectP(ray);
	}

	auto Scene::intersectTr(Ray ray, Sampler& sampler, SurfaceInteraction* isect, Spectrum* Tr) const noexcept -> bool {
		*Tr = Spectrum(1.f);
		while (true) {
			bool hitSurface = intersect(ray, isect);
			// Accumulate beam transmittance for ray segment
			if (ray.medium)
				*Tr *= ray.medium->Tr(ray, sampler);
			// Initialize next ray segment or terminate transmittance computation
			if (!hitSurface)
				return false;
			if (isect->primitive->getMaterial() != nullptr)
				return true;
			ray = isect->spawnRay(ray.d);
		}
	}
}