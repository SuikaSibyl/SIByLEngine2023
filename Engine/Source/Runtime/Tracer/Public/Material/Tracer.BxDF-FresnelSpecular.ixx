module;
export module Tracer.BxDF:FresnelSpecular;
import :BxDF;
import :Fresnel;
import :SpecularTransmission;
import SE.Math.Geometric;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/**
	* A single BxDF that presents both specular reflection & specular transmission together.
	* The relative weightings of the types of scattering are moduled by the dielectric Fresnel equations.
	*/
	export struct FresnelSpecular :public BxDF
	{
		FresnelSpecular(Spectrum const& R, Spectrum const& T, float etaA, float etaB, TransportMode mode)
			: BxDF(BxDF::Type(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR))
			, R(R), T(T), etaA(etaA), etaB(etaB), fresnel(etaA, etaB), mode(mode) {}

		/** For an arbitrary pair of directions the delta function returns no scattering */
		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			return Spectrum(0.f);
		}

		//virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi, Math::ipoint2 const& sample,
		//	float* pdf, Type* sampledType = nullptr) const noexcept -> Spectrum override {
		//	return Spectrum();
		//}
		//
		//virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi, Math::point2 const& u,
		//	float* pdf, Type* sampledType = nullptr) const noexcept -> Spectrum override {
		//	float F = frDielectric(cosTheta(wo), etaA, etaB);
		//	if (u[0] < F) {
		//		// Compute specular reflection for FresnelSpecular
		//		//  Compute perfect specular reflection direction
		//		*wi = Math::vec3(-wo.x, -wo.y, wo.z);
		//		if (sampledType)
		//			*sampledType = BxDF::Type(BSDF_SPECULAR | BSDF_REFLECTION);
		//		*pdf = F;
		//		return F* R / absCosTheta(*wi);
		//	}
		//	else {
		//		// Compute specular transmission for FresnelSpecular
		//		//  Figure out which η is incidentand which is transmitted
		//		bool entering = cosTheta(wo) > 0;
		//		float etaI = entering ? etaA : etaB;
		//		float etaT = entering ? etaB : etaA;
		//		//  Compute ray direction for specular transmission
		//		if (!refract(wo, faceforward(Math::normal3(0, 0, 1), wo), etaI / etaT, wi))
		//			return 0;
		//		Spectrum ft = T * (1 - F);
		//		//  Account for non - symmetry with transmission to different medium
		//		if (mode == TransportMode::Radiance)
		//			ft *= (etaI * etaI) / (etaT * etaT);
		//		if (sampledType)
		//			*sampledType = BxDF::Type(BSDF_SPECULAR | BSDF_TRANSMISSION);
		//		*pdf = 1 - F;
		//		return ft / absCosTheta(*wi);
		//	}
		//}
		//
	private:
		Spectrum const R, T;
		float const etaA, etaB;
		/** only focus on the dielectric case */
		FresnelDielectric const fresnel;
		TransportMode const mode;
	};
}