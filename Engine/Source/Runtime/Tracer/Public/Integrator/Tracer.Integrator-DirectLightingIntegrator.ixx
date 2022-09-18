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
}