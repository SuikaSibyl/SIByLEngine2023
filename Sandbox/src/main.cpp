#include <iostream>
#include <chrono>
#include <format>
#include <array>
#include <functional>
#include <filesystem>
#include <chrono>
#include <memory>
#include <typeinfo>
#include <glad/glad.h>
#include <imgui.h>
import Core.Log;
import Core.Memory;
import Core.IO;
import Core.Event;
import Core.Timer;
import Core.ECS;
import Core.Resource.RuntimeManage;

import Math.Vector;
import Math.Geometry;
import Math.Matrix;
import Math.Limits;
import Math.Transform;
import Math.Trigonometric;

import Tracer.Ray;
import Tracer.Camera;
import Tracer.Film;
import Tracer.Shape;
import Tracer.Filter;
import Tracer.Primitives;
import Tracer.Interactable;
import Tracer.Material;
import Tracer.Integrator;
import Tracer.Sampler;
import Tracer.Texture;
import Tracer.Light;

import Image.Color;
import Image.Image;
import Image.FileFormat;

import Platform.Window;
import Platform.System;
import Parallelism.Parallel;

import RHI;
import RHI.RHILayer;

import GFX.Resource;
import GFX.GFXManager;
import GFX.MeshLoader;
import Application.Root;
import Application.Base;

import Editor.Core;
import Editor.Framework;
import Editor.GFX;
import Editor.Config;

using namespace SIByL;
using namespace SIByL::Core;
using namespace SIByL::Math;

struct UniformBufferObject {
	Math::mat4 model;
	Math::mat4 view;
	Math::mat4 proj;
	Math::mat4 viewInverse;  // Camera inverse view matrix
	Math::mat4 projInverse;  // Camera inverse projection matrix
};

struct GlobalUniforms {
	Math::mat4 viewProj;     // Camera view * projection
	Math::mat4 viewInverse;  // Camera inverse view matrix
	Math::mat4 projInverse;  // Camera inverse projection matrix
};

struct PushConstantRay {
	Math::vec4 clearColor;
	Math::vec3 lightPosition;
	float lightIntensity;
	int   lightType;
};

struct SandBoxApplication :public Application::ApplicationBase {
	/** Initialize the application */
	virtual auto Init() noexcept -> void override {
		// create optional layers: rhi, imgui, editor
		rhiLayer = std::make_unique<RHI::RHILayer>(RHI::RHILayerDescriptor{
				RHI::RHIBackend::Vulkan,
				(uint32_t)RHI::ContextExtension::RAY_TRACING,
				mainWindow.get(),
				true
			});
		GFX::GFXManager::get()->rhiLayer = rhiLayer.get();
		imguiLayer = std::make_unique<Editor::ImGuiLayer>(rhiLayer.get());
		editorLayer = std::make_unique<Editor::EditorLayer>();
		Editor::Config::buildEditorLayer(editorLayer.get());
		// bind editor layer
		editorLayer->getWidget<Editor::SceneWidget>()->bindScene(&scene);

		RHI::Device* device = rhiLayer->getDevice();
		RHI::SwapChain* swapchain = rhiLayer->getSwapChain();


		Core::Buffer vertex_CornellBox;
		Core::Buffer vertex_CornellBox_POnly;
		Core::Buffer index_CornellBox;
		GFX::MeshLoader_OBJ::loadOBJ(
			"./content/CornellBox-Original-Merged.obj",
			GFX::MeshDataLayout{ 
				{{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::POSITION},
				{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::COLOR}},
				RHI::IndexFormat::UINT16_t },
			&vertex_CornellBox,
			&index_CornellBox
		);
		GFX::MeshLoader_OBJ::loadOBJ(
			"./content/CornellBox-Original-Merged.obj",
			GFX::MeshDataLayout{ 
				{{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::POSITION}},
				RHI::IndexFormat::UINT16_t },
			&vertex_CornellBox_POnly,
			nullptr
		);
		indexCount = index_CornellBox.size / sizeof(uint16_t);

		struct Vertex {
			Math::vec3 pos;
			Math::vec3 color;
		};

		GFX::Mesh mesh;
		mesh.vertexBuffer = device->createDeviceLocalBuffer((void*)vertex_CornellBox.data, vertex_CornellBox.size,
			(uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::STORAGE);
		mesh.indexBuffer = device->createDeviceLocalBuffer((void*)index_CornellBox.data, index_CornellBox.size,
			(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
			(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		ResourceManager::get()->addResource<GFX::Mesh>(0, std::move(mesh));
		GFX::Mesh* mesh_resource = ResourceManager::get()->getResource<GFX::Mesh>(0);

		vertexBufferRT = device->createDeviceLocalBuffer((void*)vertex_CornellBox_POnly.data, vertex_CornellBox_POnly.size,
			(uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
			(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);

		blas = device->createBLAS(RHI::BLASDescriptor{
			vertexBufferRT.get(),
			mesh_resource->indexBuffer.get(),
			(uint32_t)vertex_CornellBox_POnly.size / (sizeof(float) * 3),
			indexCount / 3,
			RHI::IndexFormat::UINT16_t });
		tlas = device->createTLAS(RHI::TLASDescriptor{ {
			{blas.get(), Math::mul(Math::translate(Math::vec3{0,0.00,0}).m, Math::scale(0.2,0.2, 0.2).m),0,0},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0,0.45,0}).m, Math::scale(0.2,0.2, 0.2).m),0,0},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0,0.90,0}).m, Math::scale(0.2,0.2, 0.2).m),0,0},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0,1.35,0}).m, Math::scale(0.2,0.2, 0.2).m),0,0},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0,1.80,0}).m, Math::scale(0.2,0.2, 0.2).m),0,0},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0.45,0.00,0}).m, Math::scale(0.2,0.2, 0.2).m),0,1},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0.45,0.45,0}).m, Math::scale(0.2,0.2, 0.2).m),0,0},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0.45,0.90,0}).m, Math::scale(0.2,0.2, 0.2).m),0,1},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0.45,1.35,0}).m, Math::scale(0.2,0.2, 0.2).m),0,0},
			{blas.get(), Math::mul(Math::translate(Math::vec3{0.45,1.80,0}).m, Math::scale(0.2,0.2, 0.2).m),0,0},
			}});

		{
			std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = Image::JPEG::fromJPEG("./content/texture.jpg");
			Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			GFX::GFXManager::get()->registerTextureResource(guid, img.get());
			GFX::Texture* texture = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);

			GFX::GFXManager::get()->commonSampler.defaultSampler = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
			GFX::GFXManager::get()->registerSamplerResource(GFX::GFXManager::get()->commonSampler.defaultSampler, RHI::SamplerDescriptor{});
			Core::ResourceManager::get()->getResource<GFX::Sampler>(GFX::GFXManager::get()->commonSampler.defaultSampler)->sampler->setName("DefaultSampler");
			//framebufferColorAttaches
			framebufferColorAttach = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			framebufferDepthAttach = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			rtTarget = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			RHI::TextureDescriptor desc{
				{800,600,1},
				1, 1, RHI::TextureDimension::TEX2D,
				RHI::TextureFormat::RGBA8_UNORM,
				(uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
				{ RHI::TextureFormat::RGBA8_UNORM }
			};
			GFX::GFXManager::get()->registerTextureResource(framebufferColorAttach, desc);
			desc.format = RHI::TextureFormat::RGBA32_FLOAT;
			desc.usage |= (uint32_t)RHI::TextureUsage::STORAGE_BINDING;
			GFX::GFXManager::get()->registerTextureResource(rtTarget, desc);
			desc.format = RHI::TextureFormat::DEPTH32_FLOAT;
			desc.usage = (uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING;
			desc.viewFormats = { RHI::TextureFormat::DEPTH32_FLOAT };
			GFX::GFXManager::get()->registerTextureResource(framebufferDepthAttach, desc);
		}
		Buffer vert, frag;
		syncReadFile("../Engine/Binaries/Runtime/spirv/Common/test_shader_vert_vert.spv", vert);
		syncReadFile("../Engine/Binaries/Runtime/spirv/Common/test_shader_frag.spv", frag);
		vert_module = device->createShaderModule({ &vert, RHI::ShaderStages::VERTEX });
		frag_module = device->createShaderModule({ &frag, RHI::ShaderStages::FRAGMENT });

		Core::GUID rgen, rmiss, mat0_rchit, mat1_rchit, comp;
		rgen = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
		rmiss = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
		mat0_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
		mat1_rchit = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
		comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
		GFX::GFXManager::get()->registerShaderModuleResource(rgen, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/raytrace_rgen.spv", { nullptr, RHI::ShaderStages::RAYGEN });
		GFX::GFXManager::get()->registerShaderModuleResource(rmiss, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/simple_sky_rmiss.spv", { nullptr, RHI::ShaderStages::MISS });
		GFX::GFXManager::get()->registerShaderModuleResource(mat0_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/diffuseMat_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
		GFX::GFXManager::get()->registerShaderModuleResource(mat1_rchit, "../Engine/Binaries/Runtime/spirv/RayTracing/RayTrace/src/specularMat_rchit.spv", { nullptr, RHI::ShaderStages::CLOSEST_HIT });
		GFX::GFXManager::get()->registerShaderModuleResource(comp, "../Engine/Binaries/Runtime/spirv/Common/test_compute_comp.spv", { nullptr, RHI::ShaderStages::COMPUTE});
		
		// create uniformBuffer
		{
			for (int i = 0; i < 2; ++i) {
				RHI::BufferDescriptor uboDescriptor;
				uboDescriptor.size = sizeof(UniformBufferObject);
				uboDescriptor.usage = (uint32_t)RHI::BufferUsage::UNIFORM;
				uboDescriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT
					| (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
				uboDescriptor.mappedAtCreation = true;
				uniformBuffer[i] = device->createBuffer(uboDescriptor);
			}
		}
		{
			bindGroupLayout = device->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, (uint32_t)RHI::ShaderStages::VERTEX | (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::UNIFORM}}
					} }
			);

			bindGroupLayout_RT = device->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT, RHI::AccelerationStructureBindingLayout{}},
					RHI::BindGroupLayoutEntry{1, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT, RHI::StorageTextureBindingLayout{}},
					RHI::BindGroupLayoutEntry{2, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
					RHI::BindGroupLayoutEntry{3, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
					} }
			);

			for (int i = 0; i < 2; ++i) {
				bindGroup[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					bindGroupLayout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{RHI::BufferBinding{uniformBuffer[i].get(), 0, uniformBuffer[i]->size()}}}
				} });
				bindGroup_RT[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					bindGroupLayout_RT.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{tlas.get()}},
						{1,RHI::BindingResource{Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->originalView.get()}},
						{2,RHI::BindingResource{{vertexBufferRT.get(), 0, vertexBufferRT.get()->size()}}},
						{3,RHI::BindingResource{{mesh_resource->indexBuffer.get(), 0, mesh_resource->indexBuffer->size()}}}
				} });
			}
		}

		for (int i = 0; i < 2; ++i) {
			pipelineLayout[i] = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{ {},
				{ bindGroupLayout.get() }
				});

			pipelineLayout_RT[i] = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
				{ {(uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::MISS | (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstantRay)}},
				{ bindGroupLayout_RT.get(), bindGroupLayout.get() }
				});

			renderPipeline[i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
				pipelineLayout[i].get(),
				RHI::VertexState{
						// vertex shader
						vert_module.get(), "main",
						// vertex attribute layout
						{ RHI::VertexBufferLayout{sizeof(Vertex), RHI::VertexStepMode::VERTEX, {
							{ RHI::VertexFormat::FLOAT32X3, 0, 0},
							{ RHI::VertexFormat::FLOAT32X3, offsetof(Vertex,color), 1},}}}},
					RHI::PrimitiveState{ RHI::PrimitiveTopology::TRIANGLE_LIST, RHI::IndexFormat::UINT16_t },
					RHI::DepthStencilState{ RHI::TextureFormat::DEPTH32_FLOAT, true, RHI::CompareFunction::LESS },
					RHI::MultisampleState{},
					RHI::FragmentState{
						// fragment shader
						frag_module.get(), "main",
						{{RHI::TextureFormat::RGBA8_UNORM}}}
					});

			raytracingPipeline[i] = device->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
				pipelineLayout_RT[i].get(),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)->shaderModule.get(),
				Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rmiss)->shaderModule.get(),
				{	Core::ResourceManager::get()->getResource<GFX::ShaderModule>(mat0_rchit)->shaderModule.get(),
					Core::ResourceManager::get()->getResource<GFX::ShaderModule>(mat1_rchit)->shaderModule.get() },
				nullptr,
				nullptr
				});

			computePipeline[i] = device->createComputePipeline(RHI::ComputePipelineDescriptor{
				pipelineLayout_RT[i].get(),
				{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp)->shaderModule.get(), "main"}
				});

			computePipeline[i]->setName("ComputeShader_RayTracer");
		}
	};

	/** Update the application every loop */
	virtual auto Update(double deltaTime) noexcept -> void override {
		// start new frame
		imguiLayer->startNewFrame();
		// frame start
		RHI::Device* device = rhiLayer->getDevice();
		RHI::SwapChain* swapChain = rhiLayer->getSwapChain();
		RHI::MultiFrameFlights* multiFrameFlights = rhiLayer->getMultiFrameFlights();
		multiFrameFlights->frameStart();

		std::unique_ptr<RHI::CommandEncoder> commandEncoder = device->createCommandEncoder({ multiFrameFlights->getCommandBuffer() });

		RHI::RenderPassDescriptor renderPassDescriptor = {
			{ RHI::RenderPassColorAttachment{
				Core::ResourceManager::get()->getResource<GFX::Texture>
				(framebufferColorAttach)->originalView.get(),
			nullptr, {0,0,0,1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE }},
			RHI::RenderPassDepthStencilAttachment{
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferDepthAttach)->originalView.get(),
				1, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false,
				0, RHI::LoadOp::CLEAR, RHI::StoreOp::DONT_CARE, false
		},
		};

		RHI::RayTracingPassDescriptor rayTracingDescriptor = {};

		uint32_t index = multiFrameFlights->getFlightIndex();

		int width, height;
		mainWindow->getFramebufferSize(&width, &height);
		width = 800;
		height = 600;

		UniformBufferObject ubo;
		Math::vec4 campos = Math::mul(Math::rotateY(0 * 20).m, Math::vec4(-0.001, 1.0, 6.0, 1));
		//ubo.model = Math::transpose(Math::rotate(timer.totalTime() * 80, Math::vec3(0, 1, 0)).m);
		ubo.view = Math::transpose(Math::lookAt(Math::vec3(campos.x, campos.y, campos.z) , Math::vec3(0, 1, 0), Math::vec3(0, 1, 0)).m);
		ubo.proj = Math::transpose(Math::perspective(22.f, 1.f * 800 / 600, 0.1f, 10.f).m);
		ubo.viewInverse = Math::inverse(ubo.view);
		//ubo.proj.data[1][1] *= -1;
		ubo.projInverse = Math::inverse(ubo.proj);
		Math::vec4 originPos = Math::vec4(0.f, 0.f, 0.f, 1.f);
		Math::mat4 invView = Math::transpose(ubo.viewInverse);
		auto originPost = Math::mul(invView, Math::vec4(0.f, 0.f, 0.f, 1.f));

		std::cout << 1.f / timer.deltaTime() << std::endl;
		//Math::rotate( )
		std::future<bool> mapped = uniformBuffer[index]->mapAsync(0, 0, sizeof(UniformBufferObject));
		if (mapped.get()) {
			void* data = uniformBuffer[index]->getMappedRange(0, sizeof(UniformBufferObject));
			memcpy(data, &ubo, sizeof(UniformBufferObject));
			uniformBuffer[index]->unmap();
		}

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferColorAttach)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				(uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
			}}
			});

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(framebufferDepthAttach)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::DEPTH_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				(uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL
			}}
			});

		{
			commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
				(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
				(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
				(uint32_t)RHI::DependencyType::NONE,
				{}, {},
				{ RHI::TextureMemoryBarrierDescriptor{
					Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->texture.get(),
					RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
					(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
					RHI::TextureLayout::GENERAL
				}}
				});

			pcRay.clearColor = Math::vec4{ 0.2f, 0.3f, 0.3f, 1.0f };
			pcRay.lightPosition = Math::vec3{ 10.f,0.f,0.f };
			pcRay.lightIntensity = 100;
			pcRay.lightType = 0;

			static uint32_t batchIdx = 0;
			rtEncoder[index] = commandEncoder->beginRayTracingPass(rayTracingDescriptor);
			rtEncoder[index]->setPipeline(raytracingPipeline[index].get());
			//rtEncoder[index]->pushConstants(&batchIdx, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::CLOSEST_HIT);
			rtEncoder[index]->setBindGroup(0, bindGroup_RT[index].get(), 0, 0);
			rtEncoder[index]->setBindGroup(1, bindGroup[index].get(), 0, 0);
			rtEncoder[index]->pushConstants(&batchIdx,
				(uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::CLOSEST_HIT
				| (uint32_t)RHI::ShaderStages::MISS | (uint32_t)RHI::ShaderStages::COMPUTE,
				0, sizeof(uint32_t));
			rtEncoder[index]->traceRays(800, 600, 1);
			rtEncoder[index]->end();
			++batchIdx;

			commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
				(uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR,
				(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
				(uint32_t)RHI::DependencyType::NONE,
				{}, {},
				{ RHI::TextureMemoryBarrierDescriptor{
					Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->texture.get(),
					RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
					(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
					(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
					RHI::TextureLayout::GENERAL,
					RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
				}}
				});
		}

		//{
		//	commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
		//		(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
		//		(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
		//		(uint32_t)RHI::DependencyType::NONE,
		//		{}, {},
		//		{ RHI::TextureMemoryBarrierDescriptor{
		//			Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->texture.get(),
		//			RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
		//			(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
		//			(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
		//			RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
		//			RHI::TextureLayout::GENERAL
		//		}}
		//		});

		//	static uint32_t batchIdx = 0;
		//	compEncoder[index] = commandEncoder->beginComputePass({});
		//	compEncoder[index]->setPipeline(computePipeline[index].get());
		//	compEncoder[index]->setBindGroup(0, bindGroup_RT[index].get(), 0, 0);
		//	compEncoder[index]->setBindGroup(1, bindGroup[index].get(), 0, 0);
		//	compEncoder[index]->pushConstants(&batchIdx, (uint32_t)RHI::ShaderStages::COMPUTE
		//		| (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::CLOSEST_HIT
		//		| (uint32_t)RHI::ShaderStages::MISS, 0, sizeof(uint32_t));
		//	compEncoder[index]->dispatchWorkgroups((800 + 15) / 16, (600 + 7) / 8, 1);
		//	compEncoder[index]->end();
		//	++batchIdx;

		//	commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
		//		(uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT,
		//		(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
		//		(uint32_t)RHI::DependencyType::NONE,
		//		{}, {},
		//		{ RHI::TextureMemoryBarrierDescriptor{
		//			Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->texture.get(),
		//			RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
		//			(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
		//			(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
		//			RHI::TextureLayout::GENERAL,
		//			RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
		//		}}
		//		});
		//}

		passEncoder[index] = commandEncoder->beginRenderPass(renderPassDescriptor);
		passEncoder[index]->setPipeline(renderPipeline[index].get());
		passEncoder[index]->setViewport(0, 0, width, height, 0, 1);
		passEncoder[index]->setScissorRect(0, 0, width, height);
		passEncoder[index]->setVertexBuffer(0, ResourceManager::get()->getResource<GFX::Mesh>(0)->vertexBuffer.get(), 
			0, ResourceManager::get()->getResource<GFX::Mesh>(0)->vertexBuffer->size());
		passEncoder[index]->setIndexBuffer(ResourceManager::get()->getResource<GFX::Mesh>(0)->indexBuffer.get(), 
			RHI::IndexFormat::UINT16_t, 0, ResourceManager::get()->getResource<GFX::Mesh>(0)->indexBuffer->size());
		passEncoder[index]->setBindGroup(0, bindGroup[index].get(), 0, 0);
		passEncoder[index]->drawIndexed(indexCount, 1, 0, 0, 0);
		passEncoder[index]->end();

		device->getGraphicsQueue()->submit({ commandEncoder->finish({}) }, 
			multiFrameFlights->getImageAvailableSeamaphore(),
			multiFrameFlights->getRenderFinishedSeamaphore(),
			multiFrameFlights->getFence());

		// GUI Recording
		imguiLayer->startGuiRecording();
		bool show_demo_window = true;
		ImGui::ShowDemoWindow(&show_demo_window);
		ImGui::Begin("Ray Tracer");
		ImGui::Image(
			Editor::TextureUtils::getImGuiTexture(rtTarget)->getTextureID(),
			{ (float)width,(float)height },
			{ 0,0 }, { 1, 1 });
		ImGui::End();
		ImGui::Begin("Rasterizer");
		ImGui::Image(
			Editor::TextureUtils::getImGuiTexture(framebufferColorAttach)->getTextureID(),
			{ (float)width,(float)height },
			{ 0,0 }, { 1, 1 });
		ImGui::End();
		editorLayer->onDrawGui();
		imguiLayer->render();


		multiFrameFlights->frameEnd();
	};
	
	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};

	virtual auto Exit() noexcept -> void override {
		rhiLayer->getDevice()->waitIdle();
		renderPipeline[0] = nullptr;
		renderPipeline[1] = nullptr;
		computePipeline[0] = nullptr;
		computePipeline[1] = nullptr;
		raytracingPipeline[0] = nullptr;
		raytracingPipeline[1] = nullptr;
		passEncoder[0] = nullptr;
		passEncoder[1] = nullptr;
		compEncoder[1] = nullptr;
		compEncoder[1] = nullptr;
		pipelineLayout[0] = nullptr;
		pipelineLayout[1] = nullptr;
		pipelineLayout_RT[0] = nullptr;
		pipelineLayout_RT[1] = nullptr;
		pipelineLayout_COMP[0] = nullptr;
		pipelineLayout_COMP[1] = nullptr;

		bindGroupLayout_RT = nullptr;
		bindGroup_RT[0] = nullptr;
		bindGroup_RT[1] = nullptr;

		vertexBufferRT = nullptr;
		indexBufferRT = nullptr;
		vert_module = nullptr;
		frag_module = nullptr;
		tlas = nullptr;
		blas = nullptr;

		bindGroupLayout = nullptr;
		uniformBuffer[0] = nullptr;
		uniformBuffer[1] = nullptr;
		bindGroup[0] = nullptr;
		bindGroup[1] = nullptr;

		editorLayer = nullptr;
		imguiLayer = nullptr;

		ResourceManager::get()->clear();

		rhiLayer = nullptr;
	}

private:
	Core::GUID framebufferColorAttach;
	Core::GUID framebufferDepthAttach;
	Core::GUID rtTarget;

	PushConstantRay pcRay = {};
	uint32_t indexCount = 0;

	std::unique_ptr<RHI::RHILayer> rhiLayer = nullptr;
	std::unique_ptr<Editor::ImGuiLayer> imguiLayer = nullptr;
	std::unique_ptr<Editor::EditorLayer> editorLayer = nullptr;

	std::unique_ptr<RHI::BindGroupLayout> bindGroupLayout = nullptr;
	std::unique_ptr<RHI::BindGroup> bindGroup[2];

	std::unique_ptr<RHI::BindGroupLayout> bindGroupLayout_RT = nullptr;
	std::unique_ptr<RHI::BindGroup> bindGroup_RT[2];

	std::unique_ptr<RHI::Buffer> uniformBuffer[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout_RT[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout_COMP[2];

	std::unique_ptr<RHI::Buffer> vertexBufferRT = nullptr;
	std::unique_ptr<RHI::Buffer> indexBufferRT = nullptr;
	std::unique_ptr<RHI::ShaderModule> vert_module = nullptr;
	std::unique_ptr<RHI::ShaderModule> frag_module = nullptr;
	std::unique_ptr<RHI::RenderPassEncoder> passEncoder[2] = {};
	std::unique_ptr<RHI::RayTracingPassEncoder> rtEncoder[2] = {};
	std::unique_ptr<RHI::ComputePassEncoder> compEncoder[2] = {};
	std::unique_ptr<RHI::BLAS> blas = nullptr;
	std::unique_ptr<RHI::TLAS> tlas = nullptr;

	std::unique_ptr<RHI::ComputePipeline> computePipeline[2];
	std::unique_ptr<RHI::RenderPipeline> renderPipeline[2];
	std::unique_ptr<RHI::RayTracingPipeline> raytracingPipeline[2];
	// the embedded scene, which should be removed in the future
	GFX::Scene scene;
};

int main()
{
	// application root, control all managers
	Application::Root root;
	// run app
	SandBoxApplication app;
	app.createMainWindow({
			Platform::WindowVendor::GLFW,
			L"SIByL Engine 2022.0",
			1280, 720,
			Platform::WindowProperties::VULKAN_CONTEX
		});
	app.run();

	//Scope<Platform::Window> window = Platform::Window::create({
	//		Platform::WindowVendor::WIN_64,
	//		L"Ray Trace Viewer",
	//		720, 480,
	//		Platform::WindowProperties::OPENGL_CONTEX
	//	});

	//int ncores = Platform::getNumSystemCores();
	//Image::Image<Image::COLOR_R8G8B8_UINT> image(720, 480);
	//std::fill((Image::COLOR_R8G8B8_UINT*) & ((reinterpret_cast<char*>(image.data.data))[0]), 
	//	(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]), 
	//	Image::COLOR_R8G8B8_UINT{ {255,255,0} });

	//window->bindPaintingBitmapRGB8(size_t(720), size_t(480), (char*)image.data.data);
	//window->resize(720.f, 480);

	//Math::Transform defaultTransform(mat4{});
	//Math::AnimatedTransform animatedDefaultTransform(&defaultTransform, 0, &defaultTransform, 0);
	//Tracer::Film film(Math::ipoint2{ 720, 480 }, Math::bounds2{ {0,0}, {1,1} }, 
	//	std::make_unique<Tracer::BoxFilter>(Math::vec2{1.f,1.f}), 1, "what.c", 1);
	//Tracer::OrthographicCamera camera(animatedDefaultTransform, 
	//	Math::bounds2{ {-1.f * 720.f / 480.f,-1.f}, {1.f * 720.f / 480.f, 1.f} } , 
	//	0, 0, 0, 0, &film, nullptr);

	//float const radius = 0.5f;
	//float const radius_2 = 3.f;
	//Math::Transform objectToWorld = Math::translate({ 0,0,radius_2 });
	//Math::Transform worldToObject = Math::translate({ 0,0,-radius_2 });
	//Tracer::Sphere sphere(&objectToWorld, &worldToObject, false, radius, -radius, radius, 360);
	//Tracer::MatteMaterial material(nullptr, nullptr, nullptr);
	//Tracer::GeometricPrimitive primitve;
	//primitve.shape = &sphere;
	//primitve.material = &material;

	//Math::Transform objectToWorld_2 = Math::translate({ 0,radius_2+ radius,radius_2 });
	//Math::Transform worldToObject_2 = Math::translate({ 0,-radius_2- radius,-radius_2 });

	//Tracer::Sphere sphere_2(&objectToWorld_2, &worldToObject_2, false, radius_2, -radius_2, radius_2, 360);
	//Tracer::GeometricPrimitive ground;
	//ground.shape = &sphere_2;
	//ground.material = &material;
	//Tracer::DummyAggregate aggregate{ std::vector<Tracer::Primitive*>{&primitve,& ground} };
	//Tracer::InfiniteAreaLight areaLight(Math::Transform{}, Tracer::Spectrum{ 1.f }, 1, "");
	//Tracer::Scene scene(&aggregate, { &areaLight });
	//Tracer::StratifiedSampler sampler(10, 10, true, 50);
	//Tracer::WhittedIntegrator integrator(5, &camera, &sampler);

	//std::thread render(std::bind(&Tracer::WhittedIntegrator::render, &integrator, scene));

	//int i = 0;
	//while (window->isRunning()) {
	//	auto startPoint = std::chrono::high_resolution_clock::now();
	//	camera.film->writeImage(image, 1.f);
	//	//// Clear background to black
	//	//std::fill((Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[0]),
	//	//	(Image::COLOR_R8G8B8_UINT*)&((reinterpret_cast<char*>(image.data.data))[image.data.size]),
	//	//	Image::COLOR_R8G8B8_UINT{ 0 ,0 ,0 });

	//	//float tHit;
	//	//Math::ivec2 sampleExtent{ 720,480 };
	//	//int const tileSize = 16;
	//	//Math::ipoint2 nTiles((sampleExtent.x + tileSize - 1) / tileSize,
	//	//	(sampleExtent.y + tileSize - 1) / tileSize);

	//	//Parallelism::ParallelFor2D([&](Math::ipoint2 tile)->void {
	//	//	Tracer::Ray ray;
	//	//	for (int i = 0; i < tileSize; ++i)
	//	//		for (int j = 0; j < tileSize; ++j) {
	//	//			image[tile.y * tileSize + j][tile.x * tileSize + i] = Image::COLOR_R8G8B8_UINT{  0,0 ,255 };
	//	//		}
	//	//}, nTiles);

	//	window->fetchEvents();
	//	window->invalid();

	//	auto endPoint = std::chrono::high_resolution_clock::now();
	//	long long start = std::chrono::time_point_cast<std::chrono::microseconds>(startPoint).time_since_epoch().count();
	//	long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endPoint).time_since_epoch().count();
	//	long long time = end - start;
	//	std::cout << "Time each frame: " << (time * 1. / 1000000) << std::endl;
	//}
	//render.join();
	//window->destroy();
	//Parallelism::clearThreadPool();
}