module;
#include <filesystem>
export module Tracer.Material:FourierMaterial;
import Core.Memory;
import Core.IO;
import Math.Common;
import Tracer.Interactable;
import Tracer.Texture;
import Tracer.Spectrum;
import Tracer.BxDF;

namespace SIByL::Tracer
{
	export struct FourierMaterial :public Material
	{
		FourierMaterial(Core::filepath const& filename, Texture<float>* bumpMap)
			: bumpMap(bumpMap) {
			FourierBSDFTable::read(filename, &bsdfTable);
		}

		virtual auto computeScatteringFunctions(
			SurfaceInteraction* si,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void override 
		{
			// Perform bump mapping with bumpMap, if present 579
			si->bsdf = Core::ArenaAlloc<BSDF>(arena, *si);
			si->bsdf->add(Core::ArenaAlloc<FourierBSDF>(arena, bsdfTable, mode));
		}

	private:
		FourierBSDFTable bsdfTable;
		Texture<float>* bumpMap;
	};
}