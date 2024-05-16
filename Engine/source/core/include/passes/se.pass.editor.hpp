#include <se.rhi.hpp>
#include <se.gfx.hpp>
#include <se.rdg.hpp>

namespace se {
  struct SIByL_API EditorInitPass :public rdg::FullScreenPass {
	EditorInitPass();
	virtual auto reflect() noexcept -> rdg::PassReflection override;
	virtual auto execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void;
  };

  struct SIByL_API BillboardPass :public rdg::RenderPass {
	BillboardPass();
	virtual auto reflect() noexcept -> rdg::PassReflection override;
	virtual auto execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void;
	auto setExternalBuffer(rhi::Buffer* buffer) noexcept -> void;
	gfx::SamplerHandle sampler;
	gfx::TextureHandle icons;
	gfx::BufferHandle billboards;
	rhi::Buffer* external_billboards = nullptr;
  };

  struct SIByL_API Line3DPass :public rdg::RenderPass {
	Line3DPass();
	virtual auto reflect() noexcept -> rdg::PassReflection override;
	virtual auto execute(rdg::RenderContext* context, rdg::RenderData const& renderData) noexcept -> void;
	gfx::BufferHandle lines;
	auto clear() noexcept -> void;
	auto addLine(se::vec3 a, se::vec3 b, se::vec3 color, float width) noexcept -> void;
	auto addAABB(se::bounds3 aabb, se::vec3 color, float width) noexcept -> void;
	auto setExternalBuffer(rhi::Buffer* buffer, size_t line_count = 0) noexcept -> void;
	rhi::Buffer* external_lines = nullptr; size_t external_count = 0;
  };
}