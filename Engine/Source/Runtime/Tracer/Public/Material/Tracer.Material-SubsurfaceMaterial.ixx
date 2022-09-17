module;
#include <new>
export module Tracer.Material:SubsurfaceMaterial;
import Core.Memory;
import Tracer.Interactable;
import Tracer.Texture;
import Tracer.Spectrum;
import Tracer.BSSRDF;

namespace SIByL::Tracer
{
	export struct SubsurfaceMaterial :public Material
	{
		virtual auto computeScatteringFunctions(
			SurfaceInteraction* si,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void override
		{
			// Perform bump mapping with bumpMap, if present 
			// Initialize BSDF for SubsurfaceMaterial
			Spectrum sig_a = scale * sigma_a->evaluate(*si).clamp();
			Spectrum sig_s = scale * sigma_s->evaluate(*si).clamp();
			si->bssrdf = Core::ArenaAlloc<TabulatedBSSRDF>(arena, *si, this, mode, eta, sig_a, sig_s, table);
		}

	private:
		float const scale;
		Texture<Spectrum>* Kr, * Kt, * sigma_a, * sigma_s;
		Texture<float>* uRoughness, * vRoughness;
		Texture<float>* bumpMap;
		float const eta;
		bool const remapRoughness;
		BSSRDFTable table;
	};
}