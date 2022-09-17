module;
#include <vector>
export module Tracer.Integrator:DirectLightingIntegrator;
import :Integrator;
import :SamplerIntegrator;
import Core.Memory;
import Math.Vector;
import Math.Geometry;
import Tracer.Ray;
import Tracer.BxDF;
import Tracer.Sampler;
import Tracer.Spectrum;
import Tracer.Sampling;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	/**
	* Different strategies for computing direct lighting, each computes an unbiased
	* estimate of exitant radiance at a point in a given direction.
	* 1. UniformSampleAll
	*	 Loops over all of the lights and takes a number of samples based on
	*	 Light::nSamples from each of them, summing the result.
	* 2. UniformSampleOne
	*	 Takes a single sample from just one of the lights, chosen at random.
	*/
	export enum struct LightStrategy {
		UniformSampleAll,
		UniformSampleOne,
	};

	/**
	* Integrator that only accounts for direct lighting: light that has traveled 
	* directly from a light source to the point being shaded. It ignores indirect
	* illumination from objects that are not emissive, except for basic specular
	* reflection and transmission effects.
	*/
	export struct DirectLightingIntegrator : public SamplerIntegrator
	{
		DirectLightingIntegrator(Camera const* camera, Sampler* sampler, LightStrategy strategy, int maxDepth)
			:SamplerIntegrator(camera, sampler), strategy(strategy), maxDepth(maxDepth) {}

		/**
		* 
		*/
		auto preprocess(Scene const& scene, Sampler& sampler) {
			if (strategy == LightStrategy::UniformSampleAll) {
				// compute number of samples to use for each light
				for (auto const& light : scene.lights)
					nLightSamples.push_back(sampler.roundCount(light->nSamples));
				// request samples for sampling all lights
				for (int i = 0; i < maxDepth; ++i) {
					for (size_t j = 0; j < scene.lights.size(); ++j) {
						sampler.request2DArray(nLightSamples[j]);
						sampler.request2DArray(nLightSamples[j]);
					}
				}
			}
		}

	private:
		/** Strategies for computing direct lighting */
		LightStrategy const strategy;
		/**
		* A maximum recursion depth for rays that are tracerd 
		* to account for specular or specular transmission. 
		*/
		int const maxDepth;

		std::vector<int> nLightSamples;
	};

	export inline auto uniformSampleOneLight(Interaction const& it, Scene const& scene,
		Core::MemoryArena& arena, Sampler& sampler, bool handleMedia) noexcept -> Spectrum {
		return Spectrum{};
	}

	export inline auto estimateDirect(Interaction const& it, Math::point2 const& uScattering, Light const& light,
		Math::point2 const& uLight, Scene const& scene, Sampler& sampler, Core::MemoryArena& arena, bool handleMedia, bool specular) {
		BxDF::Type bsdfFlags = specular ? BxDF::BSDF_ALL : BxDF::Type(BxDF::BSDF_ALL & ~BxDF::BSDF_SPECULAR);
		Spectrum Ld(0.f);
		// sample light source with multiple importance sampling
		Math::vec3 wi;
		float lightPdf = 0.f, scatteringPdf = 0.f;
		VisibilityTester visibility;
		Spectrum Li = light.sample_Li(it, uLight, &wi, &lightPdf, &visibility);
		if (lightPdf > 0.f && !Li.isBlack()) {
			// Compute BSDF or phase function's value for light sample
			Spectrum f;
			if (it.isSurfaceInteraction()) {
				// Evaluate BSDF for light sampling strategy
				SurfaceInteraction const& isect = (SurfaceInteraction const&)it;
				f = isect.bsdf->f(isect.wo, wi, bsdfFlags) * absDot(wi, isect.shading.n);
				scatteringPdf = isect.bsdf->pdf(isect.wo, wi, bsdfFlags);
			}
			else {
				// Evaluate phase function for light sampling strategy
			}
			if (!f.isBlack()) {
				// compute effect of visibility for light source sample
				if (handleMedia)
					Li *= visibility.Tr(scene, sampler);
				else if (!visibility.unoccluded(scene))
					Li = Spectrum(0.f);
				// add light's contribution to reflected radiance
				if (!Li.isBlack()) {
					if (isDeltaLight(light.flags))
						Ld += f * Li / lightPdf;
					else {
						float weight = powerHeuristic(1, lightPdf, 1, scatteringPdf);
						Ld += f * Li * weight / lightPdf;
					}
				}
			}
		}
		// sample BSDF with multiple importance sampling
		if (!isDeltaLight(light.flags)) {
			Spectrum f;
			bool sampledSpecular = false;
			if (it.isSurfaceInteraction()) {
				// Sample scattered direction for surface interactions
				BxDF::Type sampledType;
				const SurfaceInteraction& isect = (const SurfaceInteraction&)it;
				f = isect.bsdf->sample_f(isect.wo, &wi, uScattering, &scatteringPdf, bsdfFlags, &sampledType);
				f *= absDot(wi, isect.shading.n);
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
					if (lightIsect.primitive->getAreaLight() == &light) {

						//Li = lightIsect.Le(-wi);
					}
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