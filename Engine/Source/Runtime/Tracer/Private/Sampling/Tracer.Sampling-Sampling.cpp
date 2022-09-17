module;
#include <cmath>
module Tracer.Sampling:Sampling;
import Math.Random;
import Math.Vector;
import Math.Geometry;
import Math.Trigonometric;

namespace SIByL::Tracer
{
	inline auto rejectionSampleDisk(Math::RNG& rng) noexcept -> Math::point2 {
		Math::point2 p;
		do {
			p.x = 1 - 2 * rng.uniformFloat();
			p.y = 1 - 2 * rng.uniformFloat();
		} while (p.x * p.x + p.y * p.y > 1);
		return p;
	}

	inline auto uniformSampleSphere(Math::point2 const& u) noexcept -> Math::vec3 {
		float z = 1 - 2 * u[0];
		float r = std::sqrt(std::max(0.f, 1.f - z * z));
		float phi = 2 * Math::float_Pi * u[1];
		return Math::vec3(r * std::cos(phi), r * std::sin(phi), z);
	}

	inline auto uniformSpherePdf() noexcept -> float {
		return Math::float_Inv4Pi;
	}
	
	inline auto uniformSampleDisk(Math::point2 const& u) noexcept -> Math::point2 {
		float r = std::sqrt(u[0]);
		float theta = 2 * Math::float_Pi * u[1];
		return Math::point2(r * std::cos(theta), r * std::sin(theta));
	}
	
	inline auto concentricSampleDisk(Math::point2 const& u) noexcept -> Math::point2 {
		// Map uniform random numbers to[−1, 1]^2
		Math::point2 uOffset = 2.f * u - Math::vec2(1, 1);
		// Handle degeneracy at the origin
		if (uOffset.x == 0 && uOffset.y == 0)
			return Math::point2(0, 0);
		// Apply concentric mapping to point
		float theta, r;
		if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
			r = uOffset.x;
			theta = Math::float_PiOver4 * (uOffset.y / uOffset.x);
		}
		else {
			r = uOffset.y;
			theta = Math::float_PiOver2 - Math::float_PiOver4 * (uOffset.x / uOffset.y);
		}
		return r * Math::point2(std::cos(theta), std::sin(theta));
	}

	inline auto cosineSampleHemisphere(Math::point2 const& u) noexcept -> Math::vec3 {
		Math::point2 d = concentricSampleDisk(u);
		float z = std::sqrt(std::max((float)0, 1 - d.x * d.x - d.y * d.y));
		return Math::vec3(d.x, d.y, z);
	}
	
	inline auto cosineSampleHemispherePdf(float cosTheta) noexcept -> float {
		return cosTheta * Math::float_InvPi;
	}

	inline auto uniformSampleCone(Math::point2 const& u, float cosThetaMax) noexcept -> Math::vec3 {
		float cosTheta = ((float)1 - u[0]) + u[0] * cosThetaMax;
		float sinTheta = std::sqrt((float)1 - cosTheta * cosTheta);
		float phi = u[1] * 2 * Math::float_Pi;
		return Math::vec3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
	}
	
	inline auto uniformSampleHemisphere(Math::point2 const& u) noexcept -> Math::vec3 {
		float z = u[0];
		float r = std::sqrt(std::max((float)0, (float)1. - z * z));
		float phi = 2 * Math::float_Pi * u[1];
		return Math::vec3(r * std::cos(phi), r * std::sin(phi), z);
	}
	
	inline auto uniformHemispherePdf() noexcept -> float {
		return Math::float_Inv2Pi;
	}

	inline auto powerHeuristic(int nf, float fPdf, int ng, float gPdf) noexcept -> float {
		float f = nf * fPdf, g = ng * gPdf;
		return (f * f) / (f * f + g * g);
	}
}