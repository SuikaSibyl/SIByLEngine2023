module;
export module Tracer.BxDF:Lambertian;
import :BxDF;
import :Fresnel;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/**
	* Models a perfect diffuse surface that scatter incident illumination euqally in all directions.
	* Not physically plausible, but reasonably approximates some surfaces such as matte paint.
	*/
	export struct LambertianReflection :public BxDF
	{
		/** @param R: reflectance spectrum, which gives the fraction of incident light that is scattered. */
		LambertianReflection(Spectrum const& R)
			: BxDF(BxDF::Type(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {}

		/** For an arbitrary pair of directions the delta function returns no scattering */
		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			return R * Math::float_InvPi;
		}

		/** The directional-hemispherical reflectance */
		virtual auto rho(Math::vec3 const&, int, Math::point2 const*) const noexcept -> Spectrum override { return R; }

		/** The  hemispherical-hemispherical reflectance */
		virtual auto rho(int, Math::point2 const*, Math::point2 const*) const noexcept -> Spectrum override { return R; }

	private:
		Spectrum const R;
	};

	/** Models a perfect Lambertian transmission through a surface. */
	export struct LambertianTransmission :public BxDF
	{
		/** @param R: reflectance spectrum, which gives the fraction of incident light that is scattered. */
		LambertianTransmission(Spectrum const& T)
			: BxDF(BxDF::Type(BSDF_TRANSMISSION | BSDF_DIFFUSE)), T(T) {}

		/** For an arbitrary pair of directions the delta function returns no scattering */
		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			return T * Math::float_InvPi;
		}

		/** The directional-hemispherical reflectance */
		virtual auto rho(Math::vec3 const&, int, Math::point2 const*) const noexcept -> Spectrum override { return T; }

		/** The  hemispherical-hemispherical reflectance */
		virtual auto rho(int, Math::point2 const*, Math::point2 const*) const noexcept -> Spectrum override { return T; }

	private:
		Spectrum const T;
	};
}