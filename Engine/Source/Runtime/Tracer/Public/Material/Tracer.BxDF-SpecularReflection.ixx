module;
export module Tracer.BxDF:SpecularReflection;
import :BxDF;
import :Fresnel;
import SE.Math.Geometric;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/** Compute the reflected direction w_r */
	export inline auto reflect(Math::vec3 const& wo, Math::vec3 const& n) noexcept -> Math::vec3 {
		return -wo + 2 * Math::dot(wo, n) * n;
	}

	/**
	* The BRDF for physically plausible specular reflection,
	* using the Fresnel interface to compute the fraction of light that is reflected.
	*/
	export struct SpecularReflection :public BxDF
	{
		/**
		* @param R		: the spectrum to scale the reflected color
		* @param fresnel: describes dielectric or conductor fresnel properties
		*/
		SpecularReflection(Spectrum const& R, Fresnel* fresnel)
			: BxDF(BxDF::Type(BSDF_REFLECTION | BSDF_SPECULAR))
			, R(R), fresnel(fresnel) {}

		/** For an arbitrary pair of directions the delta function returns no scattering */
		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			return Spectrum(0.f);
		}

		/** 
		* Selects an appripriate direction according to the delta distribution.
		* @param wi: the output variable to be reflection of the supplied direction wo about the surface normal.
		*/
		virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi,
			Math::point2 const& sample, float* _pdf, Type* sampledType = nullptr) const noexcept -> Spectrum override {
			// compute perfect specular reflection direction
			*wi = Math::vec3(-wo.x, -wo.y, wo.z);
			*_pdf = 1.f;
			// the BRDF is based on the delta distribution, but need to divide absCosTheta
			return fresnel->evaluate(cosTheta(*wi)) * R / absCosTheta(*wi);
		}

		virtual auto pdf(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float override {
			return 0;
		}

	private:
		Spectrum const R;
		Fresnel const* fresnel;
	};
}