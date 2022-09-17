module;
#include <algorithm>
#include <cmath>
export module Tracer.BxDF:Fresnel;
import Math.Vector;
import Math.Geometry;
import Math.Common;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/*
	* The Fresnel equation describes the amount of light reflected from a surface.
	* They are the solution to Maxwell's equations at smooth surface.
	*/

	/** Computes the Fresnel reflection formula for dilectric materials & unpolarized light */
	export inline auto frDielectric(float cosThetaI, float etaI, float etaT) noexcept -> float {
		cosThetaI = Math::clamp(cosThetaI, -1.f, 1.f);
		// potentially swap indices of refraction
		bool const entering = cosThetaI > 0.f;
		if (!entering) {
			std::swap(etaI, etaT);
			cosThetaI = -cosThetaI;
		}
		// compute cosThetaT using Snell's law
		float const sinThetaI = std::sqrt(std::max(0.f, 1 - cosThetaI * cosThetaI));
		float const sinThetaT = etaI / etaT * sinThetaI;
		float const cosThetaT = std::sqrt(std::max(0.f, 1 - sinThetaT * sinThetaT));
		// handle total internal reflection (TIR)
		if (sinThetaT >= 1) return 1.f;
		// Fresnel Equation for real index of refraction
		float const Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
		float const Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
		return (Rparl * Rparl + Rperp * Rperp) / 2; // unpolarized
	}

	/** 
	* Computes the Fresnel reflection formula for conductor materials (complex IOR) and unpolarized light
	* Which actually handles dielectric-conductor case.
	* Notice both η (IOR) and k (absorption coefficient k) are wavelength-dependent data, using struct Spectrum here.
	* @ref https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
	*/
	export inline auto frConductor(float cosThetaI, Spectrum const& etaI, Spectrum const& etaT, Spectrum const& k) noexcept -> Spectrum {
		// computing thetaI information
		cosThetaI = Math::clamp(cosThetaI, -1.f, 1.f);
		float const cosThetaI2 = cosThetaI * cosThetaI;
		float const sinThetaI2 = 1.f - cosThetaI2;
		// Assuming eatI will be a dielectric, which means overline etaI will be real.
		// Terefore the relative index of refraction (= {overline etaI}/{overline etaT}) do not need a complex division,
		// and here we use normal real division instead, where overline eta = eta + etak * i.
		Spectrum const eta = etaT / etaI;
		Spectrum const etak = k / etaI;
		Spectrum const eta2 = eta * eta;
		Spectrum const etak2 = etak * etak;
		// calculate a^2 + b^2
		Spectrum const t0 = eta2 - etak2 - sinThetaI2;
		Spectrum const a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
		// calcuate Rs
		Spectrum const t1 = a2plusb2 + cosThetaI2;
		Spectrum const a = sqrt(0.5f * (a2plusb2 + t0));
		Spectrum const t2 = 2.f * cosThetaI * a;
		Spectrum const Rs = (t1 - t2) / (t1 + t2);
		// calcuate Rp
		Spectrum const t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
		Spectrum const t4 = t2 * sinThetaI2;
		Spectrum const Rp = Rs * (t3 - t4) / (t3 + t4);
		return (Rp + Rs) * 0.5f; // unpolarized
	}

	/** An interface for computing Fresnel reflection coefficients */
	export struct Fresnel {
		/** 
		* Given the cosine of the angle made by the incoming direction and the surface normal,
		* returns the amount of light reflected by the surface
		*/
		virtual auto evaluate(float cosI) const noexcept -> Spectrum = 0;
	};

	/** implements Fresnel interface for conductors */
	export struct FresnelConductor :public Fresnel {
		FresnelConductor(Spectrum const& etaI, Spectrum const& etaT, Spectrum const& k)
			:etaI(etaI), etaT(etaT), k(k) {}

		virtual auto evaluate(float cosThetaI) const noexcept -> Spectrum override {
			return frConductor(std::abs(cosThetaI), etaI, etaT, k);
		}
	private:
		Spectrum const etaI, etaT, k;
	};

	/** implements Fresnel interface for dielectric */
	export struct FresnelDielectric :public Fresnel {
		FresnelDielectric(float etaI, float etaT) :etaI(etaI), etaT(etaT) {}

		virtual auto evaluate(float cosThetaI) const noexcept -> Spectrum override {
			return frDielectric(cosThetaI, etaI, etaT);
		}
	private:
		float const etaI, etaT;
	};

	/** implementation of the Fresnel interface returns 100% reflection for all incoming directions */
	export struct FresnelNoOp : public Fresnel {
		virtual auto evaluate(float cosThetaI) const noexcept -> Spectrum override { return Spectrum(1.f); }
	};
}