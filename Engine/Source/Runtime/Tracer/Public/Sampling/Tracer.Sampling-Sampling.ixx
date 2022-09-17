export module Tracer.Sampling:Sampling;
import Math.Random;
import Math.Vector;
import Math.Geometry;
import Math.Trigonometric;

namespace SIByL::Tracer
{
	export inline auto rejectionSampleDisk(Math::RNG& rng) noexcept -> Math::point2;

	export inline auto uniformSampleSphere(Math::point2 const& u) noexcept -> Math::vec3;

	export inline auto uniformSpherePdf() noexcept -> float;

	export inline auto uniformSampleDisk(Math::point2 const& u) noexcept -> Math::point2;

	export inline auto concentricSampleDisk(Math::point2 const& u) noexcept -> Math::point2;

	export inline auto cosineSampleHemisphere(Math::point2 const& u) noexcept -> Math::vec3;

	export inline auto cosineSampleHemispherePdf(float cosTheta) noexcept -> float;

	export inline auto uniformSampleCone(Math::point2 const& u, float cosThetaMax) noexcept -> Math::vec3;

	export inline auto uniformSampleHemisphere(Math::point2 const& u) noexcept -> Math::vec3;
	
	export inline auto uniformHemispherePdf() noexcept -> float;

	export inline auto powerHeuristic(int nf, float fPdf, int ng, float gPdf) noexcept -> float;
}