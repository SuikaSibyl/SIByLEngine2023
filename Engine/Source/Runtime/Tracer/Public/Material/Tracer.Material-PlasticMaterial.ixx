export module Tracer.Material:PlasticMaterial;
import Tracer.Interactable;
import Tracer.Texture;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct PlasticMaterial :public Material
	{
		PlasticMaterial(Texture<Spectrum>* const& Kd,
			Texture<Spectrum>* const& Ks,
			Texture<float>* const& roughness,
			Texture<float>* const& bumpMap,
			bool remapRoughness)
			: Kd(Kd), Ks(Ks), roughness(roughness)
			, bumpMap(bumpMap), remapRoughness(remapRoughness) {}

	private:
		Texture<Spectrum>* Kd, *Ks;
		Texture<float>* roughness, * bumpMap;
		bool const remapRoughness;
	};
}