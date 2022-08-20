export module Tracer.Interactable:Material;
import Core.Memory;

namespace SIByL::Tracer
{
	export struct SurfaceInteraction;

	export enum struct TransportMode {
		Radiance,
		Importance,
	};

	export struct Material
	{
		virtual auto computeScatteringFunctions(
			SurfaceInteraction* isec,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void = 0;
	};
}