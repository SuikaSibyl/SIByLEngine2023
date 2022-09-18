module;
#include <cmath>
#include <vector>
export module Tracer.Integrator:Utility;
import Core.Memory;
import Math.Vector;
import Math.Geometry;
import Tracer.Ray;
import Tracer.BxDF;
import Tracer.Spectrum;
import Tracer.Interactable;
import Tracer.Sampling;

namespace SIByL::Tracer
{
	/**
	* An approach to estimate incident radiance directly from light sources,
	* by estimate each light source individually and adding the results together.
	* @param handleMedia : indicates whether the effects of volumetric attenuation should be considered
	* 
	*/
	export inline auto uniformSampleAllLight(Interaction const& it, Scene const& scene,
		Core::MemoryArena& arena, Sampler& sampler, 
		std::vector<int> const& nLightSamples, bool handleMedia) noexcept -> Spectrum;

	/**
	* An approach to estimate incident radiance directly from light sources,
	* by uniformly-randomly chose one light and multiply the result by the number of lights to compensate.
	*/
	export inline auto uniformSampleOneLight(Interaction const& it, Scene const& scene,
		Core::MemoryArena& arena, Sampler& sampler, bool handleMedia = false) noexcept -> Spectrum;

	/**
	* An approach to estimate incident radiance directly from light sources,
	* by randomly chose one light and multiply the result by the number of lights to compensate,
	* according to the total power of each light.
	*/
	export inline auto importanceSampleOneLightByTotalPower(Interaction const& it, Scene const& scene,
		Core::MemoryArena& arena, Sampler& sampler, bool handleMedia = false) noexcept -> Spectrum;

	/**
	* For one light sample, computes the value of the Monte Carlo estimator for its contribuition.
	* @param handleMedia : indicates whether the effects of volumetric attenuation should be considered
	* @param specular	 : indicates whether or not perfectly specular lobes should be considered
	*/
	export inline auto estimateDirect(
		Interaction const& it, Math::point2 const& uScattering, Light const& light,
		Math::point2 const& uLight, Scene const& scene, Sampler& sampler, 
		Core::MemoryArena& arena, bool handleMedia = false, bool specular = false) noexcept -> Spectrum;


	//////////////////////////////////////////////////////////////////////////////
	//
	//  Implementation Segment

	inline auto uniformSampleAllLight(Interaction const& it, Scene const& scene,
		Core::MemoryArena& arena, Sampler& sampler,
		std::vector<int> const& nLightSamples, bool handleMedia = false) noexcept -> Spectrum
	{
		Spectrum L(0.f);
		for (size_t j = 0; j < scene.lights.size(); ++j) {
			// Accumulate contribution of jth light to L
			Light const* light = scene.lights[j];
			int const nSamples = nLightSamples[j];
			Math::point2 const* uLightArray = sampler.get2DArray(nSamples);
			Math::point2 const* uScatteringArray = sampler.get2DArray(nSamples);
			if (!uLightArray || !uScatteringArray) {
				// Use a single sample for illumination from light
				Math::point2 const uLight = sampler.get2D();
				Math::point2 const uScattering = sampler.get2D();
				L += estimateDirect(it, uScattering, *light, uLight, scene, sampler, arena, handleMedia);
			}
			else {
				// Estimate direct lighting using sample arrays
				Spectrum Ld(0.f);
				for (int k = 0; k < nSamples; ++k)
					Ld += estimateDirect(it, uScatteringArray[k], *light, 
						uLightArray[k], scene, sampler, arena, handleMedia);
				L += Ld / nSamples;
			}
		}
		return L;
	}

	inline auto uniformSampleOneLight(Interaction const& it, Scene const& scene,
		Core::MemoryArena& arena, Sampler& sampler, bool handleMedia) noexcept -> Spectrum
	{
		// Randomly choose a single light to sample
		int const nLights = int(scene.lights.size());
		if (nLights == 0) return Spectrum(0.f);
		int const lightNum = std::min((int)(sampler.get1D() * nLights), nLights - 1);
		Light* light = scene.lights[lightNum];
		// Sample the light chosen
		Math::point2 uLight = sampler.get2D();
		Math::point2 uScattering = sampler.get2D();
		return (float)nLights * estimateDirect(it, uScattering,
			*light, uLight, scene, sampler, arena, handleMedia);
	}
	
	inline auto estimateDirect(
		Interaction const& it, Math::point2 const& uScattering, Light const& light,
		Math::point2 const& uLight, Scene const& scene, Sampler& sampler,
		Core::MemoryArena& arena, bool handleMedia, bool specular) noexcept -> Spectrum
	{
		BxDF::Type bsdfFlags = specular ? BxDF::BSDF_ALL : BxDF::Type(BxDF::BSDF_ALL & ~BxDF::BSDF_SPECULAR);
		Spectrum Ld(0.f);
		// Sample light source with multiple importance sampling
		Math::vec3 wi;
		float lightPdf = 0.f, scatteringPdf = 0.f;
		VisibilityTester visibility;
		Spectrum Li = light.sample_Li(it, uLight, &wi, &lightPdf, &visibility);
		if (lightPdf > 0 && !Li.isBlack()) {
			// Compute BSDF or phase function's value for light sample
			Spectrum f;
			if (it.isSurfaceInteraction()) {
				// Evaluate BSDF for light sampling strategy
				SurfaceInteraction const& isect = (SurfaceInteraction const&)it;
				f = isect.bsdf->f(isect.wo, wi, bsdfFlags) * Math::absDot(wi, isect.shading.n);
				scatteringPdf = isect.bsdf->pdf(isect.wo, wi, bsdfFlags);
			}
			else {
				// Evaluate phase function for light sampling strategy
			}
			if (!f.isBlack()) {
				// Compute effect of visibility for light source sample
				if (handleMedia)
					Li *= visibility.Tr(scene, sampler);
				else if (!visibility.unoccluded(scene))
					Li = Spectrum(0.f);
				// Add light's contribution to reflected radiance
				if (!Li.isBlack()) {
					if (isDeltaLight(light.flags))
						Ld += f * Li / lightPdf;
					else {
						float const weight = powerHeuristic(1, lightPdf, 1, scatteringPdf);
						Ld += f * Li * weight / lightPdf;
					}
				}
			}
		}
		// Sample BSDF with multiple importance sampling
		if (!isDeltaLight(light.flags)) {
			Spectrum f;
			bool sampledSpecular = false;
			if (it.isSurfaceInteraction()) {
				// Sample scattered direction for surface interactions
				BxDF::Type sampledType;
				const SurfaceInteraction& isect = (const SurfaceInteraction&)it;
				f = isect.bsdf->sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
					bsdfFlags, &sampledType);
				f *= Math::absDot(wi, isect.shading.n);
				sampledSpecular = sampledType & BxDF::BSDF_SPECULAR;
			}
			else {
				// Sample scattered direction for medium interactions
			}
			if (!f.isBlack() && scatteringPdf > 0) {
				// Account for light contributions along sampled direction wi
				float weight = 1;
				if (!sampledSpecular) {
					lightPdf = light.pdf_Li(it, wi);
					if (lightPdf == 0)
						return Ld;
					weight = powerHeuristic(1, scatteringPdf, 1, lightPdf);
				}
				//  Find intersectionand compute transmittance
				SurfaceInteraction lightIsect;
				Ray ray = it.spawnRay(wi);
				Spectrum Tr(1.f);
				bool foundSurfaceInteraction = handleMedia ?
					scene.intersectTr(ray, sampler, &lightIsect, &Tr) :
					scene.intersect(ray, &lightIsect);
				//  Add light contribution from material sampling
				Spectrum Li(0.f);
				if (foundSurfaceInteraction) {
					if (lightIsect.primitive->getAreaLight() == &light)
						Li = lightIsect.Le(-wi);
				}
				else
					Li = light.Le(ray);
				if (!Li.isBlack())
					Ld += f * Li * Tr * weight / scatteringPdf;
			}
		}
		return Ld;
	}
}