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
import GFX.SceneNodeLoader;
import Application.Root;
import Application.Base;

import Editor.Core;
import Editor.Framework;
import Editor.GFX;
import Editor.Config;

import Sandbox.Tracer;
import Sandbox.AAF_GI;
import Sandbox.Benchmark;

import Image.FileFormat;

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

		GFX::SceneNodeLoader_glTF::loadSceneNode("./content/scenes/cornellBox.gltf", scene);

		cornellBox = GFX::MeshLoader_OBJ::loadMeshResource(
			"./content/CornellBox-Original-Merged.obj",
			GFX::MeshDataLayout{ 
				{{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::POSITION},
				{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::COLOR}},
				RHI::IndexFormat::UINT16_t },
				true);
		grid1 = GFX::MeshLoader_OBJ::loadMeshResource(
			"./content/grids2/grid1.obj",
			GFX::MeshDataLayout{ 
				{{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::POSITION},
				{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::COLOR}},
				RHI::IndexFormat::UINT16_t },
				true);
		floor = GFX::MeshLoader_OBJ::loadMeshResource(
			"./content/grids2/floor.obj",
			GFX::MeshDataLayout{ 
				{{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::POSITION},
				{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::COLOR}},
				RHI::IndexFormat::UINT16_t },
				true);
		grid2 = GFX::MeshLoader_OBJ::loadMeshResource(
			"./content/grids2/grid2.obj",
			GFX::MeshDataLayout{ 
				{{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::POSITION},
				{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::COLOR}},
				RHI::IndexFormat::UINT16_t },
				true);
		grid3 = GFX::MeshLoader_OBJ::loadMeshResource(
			"./content/grids2/grid3.obj",
			GFX::MeshDataLayout{ 
				{{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::POSITION},
				{RHI::VertexFormat::FLOAT32X3, GFX::MeshDataLayout::VertexInfo::COLOR}},
				RHI::IndexFormat::UINT16_t },
				true);
		GFX::Mesh* conrnell_resource = ResourceManager::get()->getResource<GFX::Mesh>(cornellBox);
		GFX::Mesh* floor_mesh = ResourceManager::get()->getResource<GFX::Mesh>(floor);
		GFX::Mesh* grid1_mesh = ResourceManager::get()->getResource<GFX::Mesh>(grid1);
		GFX::Mesh* grid2_mesh = ResourceManager::get()->getResource<GFX::Mesh>(grid2);
		GFX::Mesh* grid3_mesh = ResourceManager::get()->getResource<GFX::Mesh>(grid3);
		indexCount = conrnell_resource->indexBuffer->size() / sizeof(uint16_t);

		struct Vertex {
			Math::vec3 pos;
			Math::vec3 color;
		};

		blas = device->createBLAS(RHI::BLASDescriptor{
			conrnell_resource->vertexBufferPosOnly.get(),
			conrnell_resource->indexBuffer.get(),
			(uint32_t)conrnell_resource->vertexBufferPosOnly->size() / (sizeof(float) * 3),
			indexCount / 3,
			RHI::IndexFormat::UINT16_t,
			(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION });
		
		blas_floor = device->createBLAS(RHI::BLASDescriptor{
			floor_mesh->vertexBufferPosOnly.get(),
			floor_mesh->indexBuffer.get(),
			(uint32_t)floor_mesh->vertexBufferPosOnly->size() / (sizeof(float) * 3),
			(uint32_t)floor_mesh->indexBuffer->size() / (3 * sizeof(uint16_t)),
			RHI::IndexFormat::UINT16_t,
			(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION });
		
		blas_grid1 = device->createBLAS(RHI::BLASDescriptor{
			grid1_mesh->vertexBufferPosOnly.get(),
			grid1_mesh->indexBuffer.get(),
			(uint32_t)grid1_mesh->vertexBufferPosOnly->size() / (sizeof(float) * 3),
			(uint32_t)grid1_mesh->indexBuffer->size() / (3 * sizeof(uint16_t)),
			RHI::IndexFormat::UINT16_t,
			(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION });
		
		blas_grid2 = device->createBLAS(RHI::BLASDescriptor{
			grid2_mesh->vertexBufferPosOnly.get(),
			grid2_mesh->indexBuffer.get(),
			(uint32_t)grid2_mesh->vertexBufferPosOnly->size() / (sizeof(float) * 3),
			(uint32_t)grid2_mesh->indexBuffer->size() / (3 * sizeof(uint16_t)),
			RHI::IndexFormat::UINT16_t,
			(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION });
		
		blas_grid3 = device->createBLAS(RHI::BLASDescriptor{
			grid3_mesh->vertexBufferPosOnly.get(),
			grid3_mesh->indexBuffer.get(),
			(uint32_t)grid3_mesh->vertexBufferPosOnly->size() / (sizeof(float) * 3),
			(uint32_t)grid3_mesh->indexBuffer->size() / (3 * sizeof(uint16_t)),
			RHI::IndexFormat::UINT16_t,
			(uint32_t)RHI::BLASGeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION });

		Math::mat4 floor_transform = {
			4.0f, 0.0f, 0.0f, 0.0778942f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 4.0f, 0.17478f,
			0.0f, 0.0f, 0.0f, 1.0f };

		Math::mat4 grid1_transform = {
			0.75840f, -0.465828, 0.455878, 2.18526,
			0.6232783f, 0.693876, -0.343688, 1.0795,
			-0.156223f, 0.549127, 0.821008, 1.23179,
			0.0, 0.0, 0.0, 1.0};
		
		Math::mat4 grid2_transform = {
			0.893628f, 0.105897f, 0.436135f, 0.142805,
			0.203204f, 0.770988f, -0.603561f, 1.0837,
			-0.40017f, 0.627984f, 0.667458f, 0.288514,
			0.0, 0.0, 0.0, 1.0};
		
		Math::mat4 grid3_transform = {
			0.109836f, 0.652392f, 0.74988f, -2.96444f,
			0.392525f, 0.664651f, -0.635738f, 1.86879f,
			-0.913159, 0.364174f, -0.183078f, 1.00696f,
			0.0, 0.0, 0.0, 1.0};

		ASGroup = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ASGroup>();
		GFX::GFXManager::get()->registerAsGroupResource(ASGroup,
			RHI::TLASDescriptor{ {
				{blas_floor.get(), floor_transform,0,0},
				{blas_grid1.get(), grid1_transform,0,0},
				{blas_grid2.get(), grid2_transform,0,0},
				{blas_grid3.get(), grid3_transform,0,0},
			} },
			std::vector<Core::GUID>{ floor, grid1, grid2, grid3 }
			//std::vector<Core::GUID>{ cornellBox, grid1 }
			);

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
			desc.usage |= (uint32_t)RHI::TextureUsage::STORAGE_BINDING | (uint32_t)RHI::TextureUsage::COPY_SRC;
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

		Core::GUID comp;
		comp = Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
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
			rtBindGroupLayout = device->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, (uint32_t)RHI::ShaderStages::VERTEX | (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::UNIFORM}}
					} }
			);

			bindGroupLayout_RT = device->createBindGroupLayout(
				RHI::BindGroupLayoutDescriptor{ {
					RHI::BindGroupLayoutEntry{0, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::AccelerationStructureBindingLayout{}},
					RHI::BindGroupLayoutEntry{1, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::StorageTextureBindingLayout{}},
					RHI::BindGroupLayoutEntry{2, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
					RHI::BindGroupLayoutEntry{3, (uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::COMPUTE | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::ANY_HIT, RHI::BufferBindingLayout{RHI::BufferBindingType::STORAGE}},
					} }
			);

			for (int i = 0; i < 2; ++i) {
				rtBindGroup[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					rtBindGroupLayout.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{RHI::BufferBinding{uniformBuffer[i].get(), 0, uniformBuffer[i]->size()}}}
				} });
				bindGroup_RT[i] = device->createBindGroup(RHI::BindGroupDescriptor{
					bindGroupLayout_RT.get(),
					std::vector<RHI::BindGroupEntry>{
						{0,RHI::BindingResource{Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup)->tlas.get()}},
						{1,RHI::BindingResource{Core::ResourceManager::get()->getResource<GFX::Texture>(rtTarget)->originalView.get()}},
						{2,RHI::BindingResource{{conrnell_resource->vertexBufferPosOnly.get(), 0, conrnell_resource->vertexBufferPosOnly.get()->size()}}},
						{3,RHI::BindingResource{{conrnell_resource->indexBuffer.get(), 0, conrnell_resource->indexBuffer->size()}}}
				} });
			}
		}

		for (int i = 0; i < 2; ++i) {
			pipelineLayout[i] = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{ {},
				{ rtBindGroupLayout.get() }
				});

			pipelineLayout_RT[i] = device->createPipelineLayout(RHI::PipelineLayoutDescriptor{
				{ {(uint32_t)RHI::ShaderStages::RAYGEN | (uint32_t)RHI::ShaderStages::CLOSEST_HIT | (uint32_t)RHI::ShaderStages::MISS | (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstantRay)}},
				{ bindGroupLayout_RT.get(), rtBindGroupLayout.get() }
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


			computePipeline[i] = device->createComputePipeline(RHI::ComputePipelineDescriptor{
				pipelineLayout_RT[i].get(),
				{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp)->shaderModule.get(), "main"}
				});

			computePipeline[i]->setName("ComputeShader_RayTracer");
		}

		directTracer = std::make_unique<Sandbox::DirectTracer>(rhiLayer.get(), std::array<RHI::PipelineLayout*, 2>{ pipelineLayout_RT[0].get() ,pipelineLayout_RT[1].get() });

		aafPipeline = std::make_unique<Sandbox::AAFPipeline>(rhiLayer.get(),
			Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget,
			rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
		aafGIPipeline = std::make_unique<Sandbox::AAF_GI_Pipeline>(rhiLayer.get(),
			Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget, 
			rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
		benchmarkPipeline = std::make_unique<Sandbox::Benchmark_Pipeline>(rhiLayer.get(), 
			Core::ResourceManager::get()->getResource<GFX::ASGroup>(ASGroup), rtTarget, 
			rtBindGroupLayout.get(), std::array<RHI::BindGroup*, 2>{rtBindGroup[0].get(), rtBindGroup[1].get()});
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
		//Math::vec4 campos = Math::mul(Math::rotateY(0 * 20).m, Math::vec4(-0.001, 1.0, 6.0, 1));
		////ubo.model = Math::transpose(Math::rotate(timer.totalTime() * 80, Math::vec3(0, 1, 0)).m);
		//ubo.view = Math::transpose(Math::lookAt(Math::vec3(campos.x, campos.y, campos.z) , Math::vec3(0, 1, 0), Math::vec3(0, 1, 0)).m);
		//ubo.proj = Math::transpose(Math::perspective(22.f, 1.f * 800 / 600, 0.1f, 10.f).m);
		Math::vec4 campos = Math::vec4(-4.5f, 2.5f, 5.5f, 1);
		{
			campos.x = (float)(campos.x * sin(timer.totalTime() * 1));
			campos.y = (float)(campos.y + cos(timer.totalTime() * 1 * 1.5));
			campos.z = (float)(campos.z * cos(timer.totalTime() * 1));
		}
		ubo.view = Math::transpose(Math::lookAt(Math::vec3(campos.x, campos.y, campos.z), Math::vec3(-1, 0.5f, 0), Math::vec3(0, 1, 0)).m);
		ubo.proj = Math::transpose(Math::perspective(60.f, 1.f * 800 / 600, 0.1f, 10.f).m);
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

		aafPipeline->composeCommands(commandEncoder.get(), index);
		//aafGIPipeline->composeCommands(commandEncoder.get(), index);
		//benchmarkPipeline->composeCommands(commandEncoder.get(), index);

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
		//	compEncoder[index]->setBindGroup(1, rtBindGroup[index].get(), 0, 0);
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
		passEncoder[index]->setVertexBuffer(0, ResourceManager::get()->getResource<GFX::Mesh>(cornellBox)->vertexBuffer.get(),
			0, ResourceManager::get()->getResource<GFX::Mesh>(cornellBox)->vertexBuffer->size());
		passEncoder[index]->setIndexBuffer(ResourceManager::get()->getResource<GFX::Mesh>(cornellBox)->indexBuffer.get(),
			RHI::IndexFormat::UINT16_t, 0, ResourceManager::get()->getResource<GFX::Mesh>(cornellBox)->indexBuffer->size());
		passEncoder[index]->setBindGroup(0, rtBindGroup[index].get(), 0, 0);
		passEncoder[index]->drawIndexed(ResourceManager::get()->getResource<GFX::Mesh>(cornellBox)->indexBuffer->size()/sizeof(uint16_t), 1, 0, 0, 0);
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
		if (ImGui::Button("Capture", { 200,100 })) {
			captureImage(rtTarget);
		}

		ImGui::End();
		ImGui::Begin("Rasterizer");
		ImGui::Image(
			Editor::TextureUtils::getImGuiTexture(framebufferColorAttach)->getTextureID(),
			{ (float)width,(float)height },
			{ 0,0 }, { 1, 1 });
		if (ImGui::Button("Capture Rasterizer", { 200,100 })) {
			captureImage(framebufferColorAttach);
		}
		ImGui::End();
		editorLayer->onDrawGui();
		imguiLayer->render();

		multiFrameFlights->frameEnd();
	};

	auto captureImage(Core::GUID src) noexcept -> void {
		static Core::GUID copyDst = 0;
		if (copyDst == 0) {
			copyDst = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			RHI::TextureDescriptor desc{
				{800,600,1},
				1, 1, RHI::TextureDimension::TEX2D,
				RHI::TextureFormat::RGBA32_FLOAT,
				(uint32_t)RHI::TextureUsage::COPY_DST,
				{ RHI::TextureFormat::RGBA32_FLOAT },
				true
			};
			GFX::GFXManager::get()->registerTextureResource(copyDst, desc);
		}
		rhiLayer->getDevice()->waitIdle();
		std::unique_ptr<RHI::CommandEncoder> commandEncoder = rhiLayer->getDevice()->createCommandEncoder({});
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
				RHI::TextureLayout::TRANSFER_SRC_OPTIMAL
			}}
		});
		commandEncoder->copyTextureToTexture(
			RHI::ImageCopyTexture{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get()
			},
			RHI::ImageCopyTexture{
				Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture.get()
			},
			RHI::Extend3D{ 800, 600, 1 }
		);
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(src)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::TRANSFER_READ_BIT,
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
				RHI::TextureLayout::TRANSFER_SRC_OPTIMAL,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
			}}
		});
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::PipelineStages::HOST_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture.get(),
				RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::HOST_READ_BIT,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL
			}}
			});
		rhiLayer->getDevice()->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
		rhiLayer->getDevice()->waitIdle();
		std::future<bool> mapped = Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->mapAsync(
			(uint32_t)RHI::MapMode::READ, 0, 800 * 600 * sizeof(vec4));
		if (mapped.get()) {
			void* data = Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->getMappedRange(0, 800 * 600 * sizeof(vec4));
			std::string filepath = mainWindow->saveFile("", Core::WorldTimePoint::get().to_string() + ".hdr");
			Image::HDR::writeHDR(filepath, 800, 600, 4, reinterpret_cast<float*>(data));
			Core::ResourceManager::get()->getResource<GFX::Texture>(copyDst)->texture->unmap();
		}
	}
	
	/** Update the application every fixed update timestep */
	virtual auto FixedUpdate() noexcept -> void override {

	};

	virtual auto Exit() noexcept -> void override {
		rhiLayer->getDevice()->waitIdle();

		aafPipeline = nullptr;
		aafGIPipeline = nullptr;
		benchmarkPipeline = nullptr;
		directTracer = nullptr;
		renderPipeline[0] = nullptr;
		renderPipeline[1] = nullptr;
		computePipeline[0] = nullptr;
		computePipeline[1] = nullptr;
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

		vert_module = nullptr;
		frag_module = nullptr;
		blas = nullptr;
		blas_floor = nullptr;
		blas_grid1 = nullptr;
		blas_grid2 = nullptr;
		blas_grid3 = nullptr;

		rtBindGroupLayout = nullptr;
		uniformBuffer[0] = nullptr;
		uniformBuffer[1] = nullptr;
		rtBindGroup[0] = nullptr;
		rtBindGroup[1] = nullptr;

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

	std::unique_ptr<RHI::BindGroupLayout> rtBindGroupLayout = nullptr;
	std::unique_ptr<RHI::BindGroup> rtBindGroup[2];

	std::unique_ptr<RHI::BindGroupLayout> bindGroupLayout_RT = nullptr;
	std::unique_ptr<RHI::BindGroup> bindGroup_RT[2];
	
	Core::GUID cornellBox;
	Core::GUID floor, grid1, grid2, grid3;
	Core::GUID ASGroup;

	std::unique_ptr<RHI::Buffer> uniformBuffer[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout_RT[2];
	std::unique_ptr<RHI::PipelineLayout> pipelineLayout_COMP[2];

	std::unique_ptr<RHI::ShaderModule> vert_module = nullptr;
	std::unique_ptr<RHI::ShaderModule> frag_module = nullptr;
	std::unique_ptr<RHI::RenderPassEncoder> passEncoder[2] = {};
	std::unique_ptr<RHI::ComputePassEncoder> compEncoder[2] = {};
	std::unique_ptr<RHI::BLAS> blas = nullptr;
	std::unique_ptr<RHI::BLAS> blas_floor = nullptr;
	std::unique_ptr<RHI::BLAS> blas_grid1 = nullptr;
	std::unique_ptr<RHI::BLAS> blas_grid2 = nullptr;
	std::unique_ptr<RHI::BLAS> blas_grid3 = nullptr;

	std::unique_ptr<Sandbox::DirectTracer> directTracer = nullptr;
	std::unique_ptr<Sandbox::AAFPipeline> aafPipeline = nullptr;
	std::unique_ptr<Sandbox::AAF_GI_Pipeline> aafGIPipeline = nullptr;
	std::unique_ptr<Sandbox::Benchmark_Pipeline> benchmarkPipeline = nullptr;

	std::unique_ptr<RHI::ComputePipeline> computePipeline[2];
	std::unique_ptr<RHI::RenderPipeline> renderPipeline[2];
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