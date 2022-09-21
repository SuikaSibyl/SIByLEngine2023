module;
#include <string>
export module Tracer.Light:InfiniteAreaLight;
import Math.Vector;
import Math.Geometry;
import Math.Transform;
import Math.Trigonometric;
import Tracer.Interactable;
import Tracer.Medium;
import Tracer.Spectrum;
import Tracer.Shape;

namespace SIByL::Tracer
{
	/**
	* An infinitely far away area light source that surrounds the entire scene.
	* 
	*/
	export struct InfiniteAreaLight :public Light
	{
		InfiniteAreaLight(Math::Transform const& lightToWorld, Spectrum const& L,
			int nSamples, std::string const& texmap)
			: Light((int)LightFlags::Infinite, lightToWorld, MediumInterface(), nSamples)
		{
			// Read texel data from texmap and initialize Lmap

			// Initialize sampling PDFs for infinite area light
			
		}
		
		/**
		* Infinite Area Light need scene wolrd bounds information, which is got by
		* finding the scene bounds after all of the scene geometry has been created.
		*/
		virtual auto preprocess(Scene const& scene) noexcept -> void override {
			scene.getWorldBound().boundingSphere(&worldCenter, &worldRadius);
		}

		virtual auto power() const noexcept -> Spectrum override {
			//return Math::float_Pi * worldRadius * worldRadius * Spectrum()
			return Spectrum{};
		}


	private:
		/** scene world bounding information */
		Math::point3 worldCenter;
		/** scene world bounding information */
		float worldRadius;
	};
}