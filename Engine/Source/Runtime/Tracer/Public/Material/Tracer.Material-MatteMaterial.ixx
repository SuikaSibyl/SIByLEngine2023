module;
#include <new>
export module Tracer.Material:MatteMaterial;
import Core.Memory;
import Math.Common;
import Tracer.Interactable;
import Tracer.Texture;
import Tracer.Spectrum;
import Tracer.BxDF;

namespace SIByL::Tracer
{
	export struct MatteMaterial :public Material
	{
		MatteMaterial(Texture<Spectrum>* kd,
			Texture<float>* sigma,
			Texture<float>* bumpMap)
			: Kd(kd), sigma(sigma), bumpMap(bumpMap) {}

		virtual auto computeScatteringFunctions(
			SurfaceInteraction* isec,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void override;

		Texture<Spectrum>* Kd;
		Texture<float>* sigma, *bumpMap;
	};

	auto MatteMaterial::computeScatteringFunctions(
		SurfaceInteraction* si,
		Core::MemoryArena& arena,
		TransportMode mode,
		bool allowMultipleLobes) const noexcept -> void 
	{
		// Perform bump mapping with bumpMap, if present
		//if (bumpMap)
		//	Bump(bumpMap, si);
		// Evaluate textures for MatteMaterial materialand allocate BRDF
		//si->bsdf = Core::ArenaAlloc<BSDF>(arena, *si, 1.f);
		//Spectrum r = Kd->evaluate(*si).clamp();
		//float sig = Math::clamp(sigma->evaluate(*si), 0.f, 90.f);
		//if (!r.isBlack()) {
		//	if (sig == 0)
		//		si->bsdf->add(Core::ArenaAlloc<LambertianReflection>(arena, r));
		//	else
		//		si->bsdf->add(Core::ArenaAlloc<OrenNayar>(arena, r, sig));
		//}
		
		// TODO
		si->bsdf = Core::ArenaAlloc<BSDF>(arena, *si, 1.f);
		si->bsdf->add(Core::ArenaAlloc<LambertianReflection>(arena, Spectrum{ 1.f }));
	}

}