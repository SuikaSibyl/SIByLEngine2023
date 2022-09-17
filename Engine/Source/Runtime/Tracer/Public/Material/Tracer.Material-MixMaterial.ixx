module;
#include <new>
export module Tracer.Material:MixMaterial;
import Core.Memory;
import Tracer.Interactable;
import Tracer.Texture;
import Tracer.Spectrum;
import Tracer.BxDF;

namespace SIByL::Tracer
{
	export struct MixMaterial :public Material
	{
		MixMaterial(Material* m1, Material* m2, Texture<Spectrum>* scale)
			: m1(m1), m2(m2), scale(scale) {}

		virtual auto computeScatteringFunctions(
			SurfaceInteraction* si,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void override
		{
			// Compute weightsand original BxDFs for mix material
			Spectrum s1 = scale->evaluate(*si).clamp();
			Spectrum s2 = (Spectrum(1.f) - s1).clamp();
			SurfaceInteraction si2 = *si;
			m1->computeScatteringFunctions(si, arena, mode, allowMultipleLobes);
			m2->computeScatteringFunctions(&si2, arena, mode, allowMultipleLobes);
			// Initialize si->bsdf with weighted mixture of BxDFs
			int n1 = si->bsdf->numComponents(), n2 = si2.bsdf->numComponents();
			for (int i = 0; i < n1; ++i)
				si->bsdf->bxdfs[i] = Core::ArenaAlloc<ScaledBxDF>(arena, si->bsdf->bxdfs[i], s1);
			for (int i = 0; i < n2; ++i)
				si->bsdf->add(Core::ArenaAlloc<ScaledBxDF>(arena, si2.bsdf->bxdfs[i], s2));
		}

	private:
		Material* m1, * m2;
		Texture<Spectrum>* scale;
	};
}