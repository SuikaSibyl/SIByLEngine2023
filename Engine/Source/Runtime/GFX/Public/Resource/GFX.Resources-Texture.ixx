module;
#include <memory>
export module GFX.Resource:Texture;
import Core.Resource.RuntimeManage;
import RHI;

namespace SIByL::GFX
{
	export struct Texture :public Core::Resource {
		/** ctors & rval copies */
		Texture() = default;
		Texture(Texture&& texture) = default;
		Texture(Texture const& texture) = delete;
		auto operator=(Texture && texture) -> Texture & = default;
		auto operator=(Texture const& texture) -> Texture & = delete;
		/** resrouce GUID */
		Core::GUID guid;
		/** texture */
		std::unique_ptr<RHI::Texture> texture = nullptr;
		/** texture display view*/
		std::unique_ptr<RHI::TextureView> originalView = nullptr;
	};
}