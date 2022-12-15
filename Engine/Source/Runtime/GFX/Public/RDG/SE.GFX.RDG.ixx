module;
#include <string>
#include <functional>
export module SE.GFX.RDG;
import SE.RHI;
import SE.GFX.Core;

namespace SIByL::GFX 
{
	export enum struct RDGPassFlag {
		NONE			= 0 << 0,
		RASTER			= 1 << 0,
		COMPUTE			= 1 << 1,
		ASYNC_COMPUTE	= 1 << 2,
	};

	export struct RDGPassNode {
		/** pass name */
		char const* name;
		/** pass flags */
		RDGPassFlag flags;
	};

	export struct RDGResourceNode {
		RDGResourceNode(RDGResourceNode const&) = delete;
		virtual ~RDGResourceNode() = default;
	};

	export struct RDGTexture :public RDGResourceNode {
		/** devirtualized/external resource */
		GFX::Texture* texture = nullptr;
	};

	export struct RDGUniformBuffer :public RDGResourceNode {
		/** devirtualized/external resource */
		GFX::Buffer* buffer = nullptr;
	};

	/** Provide resources for pass execute phase */
	export struct RDGRegistry {
		
	};

	export struct RDGraph {
		////////////////////////////////////
		//  Setup Phase
		// -------------------------------
		/** register external resource - texture */
		auto registerExternalTexture() noexcept -> RDGTexture*;
		/** create resource - texture */
		auto createTexture(char const* name) noexcept -> RDGTexture*;

		/** a custom pass execution */
		using CustomPassExecuteFn = std::function<void(RDGRegistry const&, RHI::CommandEncoder*)>;
		/** a custom pass setup, should return a custom pass execution */
		using CustomPassSetupFn = std::function<CustomPassExecuteFn()>;
		/** add a pass to the RDG */
		auto addPass(char const* pass_name, RDGPassFlag flag, CustomPassSetupFn const& custom_setup) noexcept -> RDGPassNode*;
		////////////////////////////////////
		//  Compile Phase
		// -------------------------------
		/** compile */
		auto compile() noexcept -> void;
		////////////////////////////////////
		//  Excute Phase
		// -------------------------------
		/** execute */
		auto execute() noexcept -> void;
		////////////////////////////////////
		//  Others
		// -------------------------------
		/** clear */
		auto clear() noexcept -> void;
	};
}