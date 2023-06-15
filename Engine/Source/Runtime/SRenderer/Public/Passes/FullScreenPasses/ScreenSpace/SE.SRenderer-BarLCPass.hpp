#pragma once

#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include <algorithm>
#include <imgui.h>
#include <imgui_internal.h>
#include "../../../../../Application/Public/SE.Application.config.h"
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>
#include <SE.Math.Geometric.hpp>
#include <Resource/SE.Core.Resource.hpp>
import SE.Platform.Window;

namespace SIByL
{
	SE_EXPORT struct BarLCPass :public RDG::FullScreenPass {

		BarLCPass();

		virtual auto reflect() noexcept -> RDG::PassReflection;

		struct alignas(64) Uniform {
			Math::vec2	view_size;
			int			hiz_mip_levels;
			uint32_t	max_iteration = 100;
			int			strategy = 0;
			int			sample_batch;
			uint32_t	debug_ray_mode = 0;
			float		max_thickness = 0.001;
			uint32_t	debug_mode = 0;
			int32_t		mip_level = 2;
			int32_t		offset_steps = 2;
			float		z_clamper = 1.0;

			Math::vec4  debugPos;
			float		z_min = 0.00211;
			float		z_range = 0.036;
			int  		is_depth = 0;
			int			lightcut_mode = 0;

			Math::mat4	InvProjMat;
			Math::mat4	ProjMat;
			Math::mat4	TransInvViewMat;
		} pConst;

		GFX::StructuredUniformBufferView<Uniform> uniform_buffer;

		virtual auto renderUI() noexcept -> void override;

		virtual auto onInteraction(
                    Platform::Input* input,
                    Editor::Widget::WidgetInfo* info) noexcept -> void override;

		virtual auto execute(RDG::RenderContext* context,
                                     RDG::RenderData const& renderData) noexcept
                    -> void;

		Core::GUID vert, frag;
		GFX::Sampler* hi_lumin_sampler = nullptr;
		GFX::Sampler* hi_z_sampler = nullptr;
		GFX::Sampler* basecolor_sampler = nullptr;
	};
}