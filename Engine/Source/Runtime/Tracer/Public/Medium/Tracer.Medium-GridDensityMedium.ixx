module;
#include <memory>
export module Tracer.Medium:HomogeneousMedium;
import SE.Core.Memory;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Ray;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct GridDensityMedium :public Medium
	{
		GridDensityMedium(Spectrum const& sigma_a, Spectrum const& sigma_s,
			float g, int nx, int ny, int nz, Math::Transform const& mediumToWorld, float const* d)
			: sigma_a(sigma_a), sigma_s(sigma_s), g(g), nx(nx), ny(ny), nz(nz)
			, worldToMedium(inverse(mediumToWorld)), density(new float[nx * ny * nz])
		{
			memcpy((float*)density.get(), d, sizeof(float) * nx * ny * nz);
			// Precompute values for Monte Carlo sampling of GridDensityMedium 896
		}

		auto getDensity(Math::point3 const& p) const noexcept -> float {
			// Compute voxel coordinatesand offsets for p
			Math::point3 pSamples(p.x * nx - .5f, p.y * ny - .5f, p.z * nz - .5f);
			Math::ipoint3 pi = (Math::ipoint3)floor(pSamples);
			Math::vec3 d = pSamples - (Math::point3)pi;
			// Trilinearly interpolate density values to compute local density
			float d00 = Math::lerp(d.x, D(pi), D(pi + Math::ivec3(1, 0, 0)));
			float d10 = Math::lerp(d.x, D(pi + Math::ivec3(0, 1, 0)), D(pi + Math::ivec3(1, 1, 0)));
			float d01 = Math::lerp(d.x, D(pi + Math::ivec3(0, 0, 1)), D(pi + Math::ivec3(1, 0, 1)));
			float d11 = Math::lerp(d.x, D(pi + Math::ivec3(0, 1, 1)), D(pi + Math::ivec3(1, 1, 1)));
			float d0 = Math::lerp(d.y, d00, d10);
			float d1 = Math::lerp(d.y, d01, d11);
			return Math::lerp(d.z, d0, d1);
		}

		auto D(Math::ipoint3 const& p) const noexcept -> float {
			Math::ibounds3 sampleBounds(Math::ipoint3(0, 0, 0), Math::ipoint3(nx, ny, nz));
			if (!insideExclusive(p, sampleBounds))
				return 0;
			return density[(p.z * ny + p.y) * nx + p.x];
		}

	private:
		Spectrum const sigma_a, sigma_s;
		float const g;
		int const nx, ny, nz;
		Math::Transform const worldToMedium;
		std::unique_ptr<float[]> density;
	};
}