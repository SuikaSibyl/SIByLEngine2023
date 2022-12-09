module;
#include <string>
export module Tracer.Light:InfiniteAreaLight;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Interactable;
import Tracer.Medium;
import Tracer.Spectrum;
import Tracer.Shape;
import Tracer.Sampling;

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

		virtual auto Le(RayDifferential const& ray) const noexcept -> Spectrum override { 
			Math::vec3 w = Math::normalize(worldToLight * ray.d);
			Math::point2 st(Math::sphericalPhi(w) * Math::float_Inv2Pi, Math::sphericalTheta(w) * Math::float_InvPi);
			return Spectrum{ 1.f };
		}
		
		virtual auto sample_Li(Interaction const& ref, Math::point2 const& u,
			Math::vec3* wi, float* pdf, VisibilityTester* vis) const noexcept -> Spectrum 
		{
			*pdf = Math::float_Inv2Pi;
			*wi = uniformSampleSphere(u);
			*vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius), ref.time, mediumInterface));
			return Spectrum{ 1.f };
		}

		virtual auto pdf_Li(Interaction const& ref, Math::vec3 const& wi) const noexcept -> float {
			return 1.f;
		}

		virtual auto sample_Le() const noexcept -> Spectrum override {
			return Spectrum{ 1.f };
		}

	private:
		/** scene world bounding information */
		Math::point3 worldCenter;
		/** scene world bounding information */
		float worldRadius;
	};
}