module;
#include <memory>
#include <vector>
#include <filesystem>
export module SE.Editor.DebugDraw;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

namespace SIByL::Editor
{
	struct DebugDrawData {
		uint32_t width, height;
	};

	struct Data_Line2D {
		struct Point {
			Math::vec2 begin;
			Math::vec2 end;
		};
		struct LineProperty {
			Math::vec3	color;
			float		width;
		};
		std::vector<Point> vertices_cpu;
		std::vector<LineProperty> lineProps_cpu;
	};

	struct Data_Line3D {
		struct Point {
			Math::vec3 begin;
			Math::vec3 end;
		};
		struct LineProperty {
			Math::vec3	color;
			float		width;
		};
		std::vector<Point> vertices_cpu;
		std::vector<LineProperty> lineProps_cpu;
	};

	export struct DrawLine2DPass :public RDG::RenderPass {

		DrawLine2DPass(Data_Line2D* data) :line2DData(data) {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/Editor/draw_line_2d/draw_line_2d_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/Editor/draw_line_2d/draw_line_2d_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::RenderPass::init(
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addInputOutput("Target")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
			.setAttachmentLoc(0));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			updateBuffer(context->flightIdx);

			GFX::Texture* color = renderData.getTexture("Target");

			renderPassDescriptor = {
				{ RHI::RenderPassColorAttachment{
					color->getRTV(0, 0, 1),
					nullptr, {0,0,0,1}, RHI::LoadOp::LOAD, RHI::StoreOp::STORE }},
				RHI::RenderPassDepthStencilAttachment{},
			};

			getBindGroup(context, 0)->updateBinding(std::vector<RHI::BindGroupEntry>{
				{ 0, RHI::BindingResource{ curr_points.getBufferBinding(context->flightIdx) }}, 
				{ 1, RHI::BindingResource{ curr_lines.getBufferBinding(context->flightIdx) } }});

			RHI::RenderPassEncoder* encoder = beginPass(context, color);

			Math::mat4 ortho = Math::transpose(Math::ortho(0, color->texture->width(), float(color->texture->height()), 0, 0, 100).m);
			encoder->pushConstants(&ortho, uint32_t(RHI::ShaderStages::VERTEX), 0, sizeof(Math::mat4));

			if (line2DData->lineProps_cpu.size() != 0)
				encoder->draw(6, line2DData->lineProps_cpu.size(), 0, 0);

			encoder->end();
		}

	private:
		auto updateBuffer(size_t idx) noexcept -> void {
			// recreate buffer
			if (curr_lines.size == 0) {
				discardGarbage();
				curr_points = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<Data_Line2D::Point>(32);
				curr_lines = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<Data_Line2D::LineProperty>(32);
			}
			if (line2DData->lineProps_cpu.size() > curr_lines.size) {
				discardGarbage();
				curr_points = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<Data_Line2D::Point>(line2DData->lineProps_cpu.size());
				curr_lines = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<Data_Line2D::LineProperty>(line2DData->lineProps_cpu.size());
			}
			// Update data
			curr_points.setStructure(line2DData->vertices_cpu.data(), idx, line2DData->vertices_cpu.size());
			curr_lines.setStructure(line2DData->lineProps_cpu.data(), idx, line2DData->lineProps_cpu.size());
		}

		auto discardGarbage() noexcept -> void {
			garbageStation.count_down = 1;
			garbageStation.points = curr_points;
			garbageStation.lines = curr_lines;
		}

		auto scanGarbageStation() noexcept -> void {
			if (garbageStation.count_down == 1) {
				if (garbageStation.points.buffer) garbageStation.points.buffer->release();
				if (garbageStation.lines.buffer) garbageStation.lines.buffer->release();
				garbageStation.count_down = -1;
			}
		}

		Data_Line2D* line2DData;
		Core::GUID vert, frag;

		struct GarbageStation {
			GFX::StructuredArrayMultiStorageBufferView<Data_Line2D::Point>			points;
			GFX::StructuredArrayMultiStorageBufferView<Data_Line2D::LineProperty>	lines;
			uint32_t count_down = -1;
		} garbageStation;
		GFX::StructuredArrayMultiStorageBufferView<Data_Line2D::Point>			curr_points;
		GFX::StructuredArrayMultiStorageBufferView<Data_Line2D::LineProperty>	curr_lines;
	};
	
	export struct DrawLine3DPass :public RDG::RenderPass {

		DrawLine3DPass(Data_Line3D* data) :line3DData(data) {
			vert = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/Editor/draw_line_3d/draw_line_3d_vert.spv", { nullptr, RHI::ShaderStages::VERTEX });
			frag = GFX::GFXManager::get()->registerShaderModuleResource("../Engine/Binaries/Runtime/spirv/Editor/draw_line_3d/draw_line_3d_frag.spv", { nullptr, RHI::ShaderStages::FRAGMENT });
			RDG::RenderPass::init(
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
		}

		virtual auto reflect() noexcept -> RDG::PassReflection {
			RDG::PassReflection reflector;

			reflector.addInputOutput("Target")
				.isTexture()
				.withSize(Math::vec3(1, 1, 1))
				.withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
				.consume(RDG::TextureInfo::ConsumeEntry{ RDG::TextureInfo::ConsumeType::ColorAttachment }
			.setAttachmentLoc(0));

			return reflector;
		}

		virtual auto execute(RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept -> void {

			updateBuffer(context->flightIdx);

			GFX::Texture* color = renderData.getTexture("Target");

			renderPassDescriptor = {
				{ RHI::RenderPassColorAttachment{
					color->getRTV(0, 0, 1),
					nullptr, {0,0,0,1}, RHI::LoadOp::LOAD, RHI::StoreOp::STORE }},
				RHI::RenderPassDepthStencilAttachment{},
			};

			getBindGroup(context, 0)->updateBinding(std::vector<RHI::BindGroupEntry>{
				{ 0, RHI::BindingResource{ curr_points.getBufferBinding(context->flightIdx) }}, 
				{ 1, RHI::BindingResource{ curr_lines.getBufferBinding(context->flightIdx) } }});

			RHI::RenderPassEncoder* encoder = beginPass(context, color);

			struct PushConstants {
				Math::mat4  projection_view;
				Math::ivec2 resolution;
			};
			PushConstants pushConstant;
			pushConstant.projection_view = renderData.getMat4("ViewProj");
			pushConstant.resolution = { int(color->texture->width()), int(color->texture->height())};
			encoder->pushConstants(&pushConstant, uint32_t(RHI::ShaderStages::VERTEX), 0, sizeof(PushConstants));

			if (line3DData->lineProps_cpu.size() != 0)
				encoder->draw(30, line3DData->lineProps_cpu.size(), 0, 0);

			encoder->end();
		}

	private:
		auto updateBuffer(size_t idx) noexcept -> void {
			// recreate buffer
			if (curr_lines.size == 0) {
				discardGarbage();
				curr_points = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<Data_Line3D::Point>(32);
				curr_lines = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<Data_Line3D::LineProperty>(32);
			}
			if (line3DData->lineProps_cpu.size() > curr_lines.size) {
				discardGarbage();
				curr_points = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<Data_Line3D::Point>(line3DData->lineProps_cpu.size());
				curr_lines = GFX::GFXManager::get()->createStructuredArrayMultiStorageBuffer<Data_Line3D::LineProperty>(line3DData->lineProps_cpu.size());
			}
			// Update data
			curr_points.setStructure(line3DData->vertices_cpu.data(), idx, line3DData->vertices_cpu.size());
			curr_lines.setStructure(line3DData->lineProps_cpu.data(), idx, line3DData->lineProps_cpu.size());
		}

		auto discardGarbage() noexcept -> void {
			garbageStation.count_down = 1;
			garbageStation.points = curr_points;
			garbageStation.lines = curr_lines;
		}

		auto scanGarbageStation() noexcept -> void {
			if (garbageStation.count_down == 1) {
				if (garbageStation.points.buffer) garbageStation.points.buffer->release();
				if (garbageStation.lines.buffer) garbageStation.lines.buffer->release();
				garbageStation.count_down = -1;
			}
		}

		Data_Line3D* line3DData;
		Core::GUID vert, frag;

		struct GarbageStation {
			GFX::StructuredArrayMultiStorageBufferView<Data_Line3D::Point>			points;
			GFX::StructuredArrayMultiStorageBufferView<Data_Line3D::LineProperty>	lines;
			uint32_t count_down = -1;
		} garbageStation;
		GFX::StructuredArrayMultiStorageBufferView<Data_Line3D::Point>			curr_points;
		GFX::StructuredArrayMultiStorageBufferView<Data_Line3D::LineProperty>	curr_lines;
	};

	export struct DebugDrawDummy :public RDG::DummyPass {
		DebugDrawDummy() {
			RDG::Pass::init();
		}

		virtual auto reflect() noexcept -> RDG::PassReflection override {
			RDG::PassReflection reflector;
			reflector.addOutput("Target")
				.isTexture();
			reflector.addOutput("Depth")
				.isTexture();
			return reflector;
		}
	};

	struct DebugDraw;

	export struct DebugDrawGraph :public RDG::Graph {
		DebugDrawGraph();
		auto setSource(GFX::Texture* ref, GFX::Texture* depth) noexcept -> void {
			setExternal("Dummy Pass", "Target", ref);
			//setExternal("Dummy Pass", "Depth", depth);
		}
	};

	export struct DebugDrawPipeline :public RDG::SingleGraphPipeline {
		DebugDrawPipeline() { pGraph = &graph; }
		DebugDrawGraph graph;
	};

	export struct DebugDraw {

		static auto Init(GFX::Texture* ref, GFX::Texture* depth) noexcept -> void;

		static auto Draw(RHI::CommandEncoder* encoder) noexcept -> void;

		static auto Clear() noexcept -> void;

		static auto DrawLine2D(Math::vec2 const& a, Math::vec2 const& b, Math::vec3 color = { 1., 0., 0. }, float width = 1.f) noexcept -> void;
		static auto DrawLine3D(Math::vec3 const& a, Math::vec3 const& b, Math::vec3 color, float width) noexcept -> void;
		static auto DrawAABB(Math::bounds3 const& aann, Math::vec3 color, float width) noexcept -> void;

		static auto Destroy() noexcept -> void;

		Data_Line2D drawLine2D;
		Data_Line3D drawLine3D;
		std::unique_ptr<DebugDrawPipeline> pipeline = nullptr;

		static auto get() -> DebugDraw* {
			if (singleton == nullptr)
				singleton = new DebugDraw();
			return singleton;
		}
		static DebugDraw* singleton;
	};

	DebugDraw* DebugDraw::singleton = nullptr;

#pragma region IMPL_DEBUG_DRAW

	DebugDrawGraph::DebugDrawGraph() {
		addPass(std::make_unique<DebugDrawDummy>(), "Dummy Pass");
		addPass(std::make_unique<DrawLine2DPass>(&DebugDraw::get()->drawLine2D), "DrawLine_2D Pass");
		addPass(std::make_unique<DrawLine3DPass>(&DebugDraw::get()->drawLine3D), "DrawLine_3D Pass");

		addEdge("Dummy Pass", "Target", "DrawLine_2D Pass", "Target");
		addEdge("DrawLine_2D Pass", "Target", "DrawLine_3D Pass", "Target");

		markOutput("DrawLine_3D Pass", "Target");
	}

	auto DebugDraw::Init(GFX::Texture* ref, GFX::Texture* depth) noexcept -> void {
		get()->pipeline = std::make_unique<DebugDrawPipeline>();
		get()->pipeline->graph.setSource(ref, depth);
		get()->pipeline->build();
	}

	auto DebugDraw::Draw(RHI::CommandEncoder* encoder) noexcept -> void {
		get()->pipeline->execute(encoder);
	}

	auto DebugDraw::Clear() noexcept -> void {
		get()->drawLine2D.vertices_cpu.clear();
		get()->drawLine2D.lineProps_cpu.clear();
		get()->drawLine3D.vertices_cpu.clear();
		get()->drawLine3D.lineProps_cpu.clear();
	}

	auto DebugDraw::DrawLine2D(Math::vec2 const& a, Math::vec2 const& b, Math::vec3 color, float width) noexcept -> void {
		get()->drawLine2D.vertices_cpu.emplace_back(Data_Line2D::Point{ a,b });
		get()->drawLine2D.lineProps_cpu.emplace_back(Data_Line2D::LineProperty{ color,width });
	}
	
	auto DebugDraw::DrawLine3D(Math::vec3 const& a, Math::vec3 const& b, Math::vec3 color, float width) noexcept -> void {
		get()->drawLine3D.vertices_cpu.emplace_back(Data_Line3D::Point{ a,b });
		get()->drawLine3D.lineProps_cpu.emplace_back(Data_Line3D::LineProperty{ color,width });
	}
	
	auto DebugDraw::DrawAABB(Math::bounds3 const& aabb, Math::vec3 color, float width) noexcept -> void {
		DrawLine3D(aabb.corner(0), aabb.corner(1), color, width);
		DrawLine3D(aabb.corner(0), aabb.corner(2), color, width);
		DrawLine3D(aabb.corner(0), aabb.corner(4), color, width);
		DrawLine3D(aabb.corner(7), aabb.corner(3), color, width);
		DrawLine3D(aabb.corner(7), aabb.corner(5), color, width);
		DrawLine3D(aabb.corner(7), aabb.corner(6), color, width);
		DrawLine3D(aabb.corner(4), aabb.corner(6), color, width);
		DrawLine3D(aabb.corner(1), aabb.corner(5), color, width);
		DrawLine3D(aabb.corner(2), aabb.corner(3), color, width);
		DrawLine3D(aabb.corner(2), aabb.corner(6), color, width);
		DrawLine3D(aabb.corner(4), aabb.corner(5), color, width);
		DrawLine3D(aabb.corner(1), aabb.corner(3), color, width);
	}

	auto DebugDraw::Destroy() noexcept -> void {
		if (singleton)
			delete singleton;
	}

#pragma endregion

}