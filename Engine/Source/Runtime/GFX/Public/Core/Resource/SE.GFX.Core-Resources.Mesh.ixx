module;
#include <vector>
#include <string>
#include <format>
#include <memory>
#include <utility>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
export module SE.GFX.Core:Mesh;
import SE.Core.Log;
import SE.Core.Resource;
import SE.Core.Memory;
import SE.Core.IO;
import SE.RHI;

namespace SIByL::GFX
{
	export struct Mesh :public Core::Resource {
		/** ctors & rval copies */
		Mesh() = default;
		Mesh(Mesh&& mesh) = default;
		Mesh(Mesh const& mesh) = delete;
		auto operator=(Mesh && mesh) -> Mesh& = default;
		auto operator=(Mesh const& mesh) -> Mesh& = delete;
		/* vertex buffer layout */
		RHI::VertexBufferLayout vertexBufferLayout = {};
		/** primitive state */
		RHI::PrimitiveState primitiveState = {};
		/** the gpu vertex buffer */
		std::unique_ptr<RHI::Buffer> vertexBuffer = nullptr;
		/** the gpu index buffer */
		std::unique_ptr<RHI::Buffer> indexBuffer = nullptr;
		/** the gpu vertex buffer with only position for ray tracing */
		std::unique_ptr<RHI::Buffer> vertexBufferPosOnly = nullptr;
		/** binded ORID */
		Core::ORID ORID = Core::ORID_NONE;
		/** submeshes */
		struct Submesh {
			uint32_t offset;
			uint32_t size;
			uint32_t baseVertex;
			uint32_t matID;
		};
		std::vector<Submesh> submeshes;
		/** resource name */
		std::string name = "New Mesh";
		/** serialize */
		inline auto serialize() noexcept -> void;
		/** deserialize */
		inline auto deserialize(RHI::Device* device, Core::ORID orid) noexcept -> void;
		/** get name */
		virtual auto getName() const noexcept -> char const* {
			return name.c_str();
		}
	};

	inline auto Mesh::serialize() noexcept -> void {
		if (ORID == Core::ORID_NONE) {
			ORID = Core::requestORID();
		}
		std::filesystem::path metadata_path = "./bin/" + std::to_string(ORID) + ".meta";
		std::filesystem::path bindata_path = "./bin/" + std::to_string(ORID) + ".bin";
		// handle metadata
		{
			YAML::Emitter out;
			out << YAML::BeginMap;
			// output type
			out << YAML::Key << "ResourceType" << YAML::Value << "Mesh";
			out << YAML::Key << "Name" << YAML::Value << name;
			out << YAML::Key << "ORID" << YAML::Value << ORID;
			// output VertexBufferLayout
			out << YAML::Key << "VertexBufferLayout" << YAML::Value;
			out << YAML::BeginMap;
			{
				out << YAML::Key << "ArrayStride" << YAML::Value << vertexBufferLayout.arrayStride;
				out << YAML::Key << "VertexStepMode" << YAML::Value << (uint32_t)vertexBufferLayout.stepMode;
				out << YAML::Key << "VertexAttributes" << YAML::Value << YAML::BeginSeq;
				for (int i = 0; i < vertexBufferLayout.attributes.size(); i++) {
					out << YAML::BeginMap;
					out << YAML::Key << "VertexFormat" << YAML::Value << (uint32_t)vertexBufferLayout.attributes[i].format;
					out << YAML::Key << "Offset" << YAML::Value << vertexBufferLayout.attributes[i].offset;
					out << YAML::Key << "Location" << YAML::Value << vertexBufferLayout.attributes[i].shaderLocation;
					out << YAML::EndMap;
				}
				out << YAML::EndSeq;
			}
			out << YAML::EndMap;
			// output PrimitiveState
			out << YAML::Key << "PrimitiveState" << YAML::Value;
			out << YAML::BeginMap;
			{					
				out << YAML::Key << "PrimitiveTopology" << YAML::Value << (uint32_t)primitiveState.topology;
				out << YAML::Key << "IndexFormat" << YAML::Value << (uint32_t)primitiveState.stripIndexFormat;
				out << YAML::Key << "FrontFace" << YAML::Value << (uint32_t)primitiveState.frontFace;
				out << YAML::Key << "CullMode" << YAML::Value << (uint32_t)primitiveState.cullMode;
				out << YAML::Key << "UnclippedDepth" << YAML::Value << primitiveState.unclippedDepth;
			}
			out << YAML::EndMap;

			out << YAML::Key << "VertexBufferSize" << YAML::Value << vertexBuffer->size();
			out << YAML::Key << "IndexBufferSize" << YAML::Value << indexBuffer->size();
			out << YAML::Key << "PosOnlyBufferSize" << YAML::Value << (vertexBufferPosOnly ? vertexBufferPosOnly->size() : 0);
			// output submeshes
			out << YAML::Key << "Submeshes" << YAML::Value << YAML::BeginSeq;
			for (int i = 0; i < submeshes.size(); i++) {
				out << YAML::BeginMap;
				out << YAML::Key << "BaseVertex" << YAML::Value << submeshes[i].baseVertex;
				out << YAML::Key << "Offset" << YAML::Value << submeshes[i].offset;
				out << YAML::Key << "Size" << YAML::Value << submeshes[i].size;
				out << YAML::Key << "MatID" << YAML::Value << submeshes[i].matID;
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;
			// output tail
			out << YAML::Key << "End" << YAML::Value << "TRUE";
			out << YAML::EndMap;
			Core::Buffer scene_proxy;
			scene_proxy.data = (void*)out.c_str();
			scene_proxy.size = out.size();
			Core::syncWriteFile(metadata_path, scene_proxy);
			scene_proxy.data = nullptr;
		}
		// handle binary data
		int vbsize = vertexBuffer->size();
		int ibsize = indexBuffer->size();
		int pbsize = (vertexBufferPosOnly ? vertexBufferPosOnly->size() : 0);
		Core::Buffer mergedBuffer(vbsize + ibsize + pbsize);
		vertexBuffer->getDevice()->waitIdle();
		vertexBuffer->getDevice()->readbackDeviceLocalBuffer(vertexBuffer.get(), mergedBuffer.data, vbsize);
		indexBuffer->getDevice()->readbackDeviceLocalBuffer(indexBuffer.get(), &(((char*)(mergedBuffer.data))[vbsize]), ibsize);
		if (vertexBufferPosOnly)
			vertexBufferPosOnly->getDevice()->readbackDeviceLocalBuffer(vertexBufferPosOnly.get(), &(((char*)(mergedBuffer.data))[vbsize + ibsize]), pbsize);
		Core::syncWriteFile(bindata_path, mergedBuffer);
	}

	inline auto Mesh::deserialize(RHI::Device* device, Core::ORID orid) noexcept -> void {
		ORID = orid;
		std::filesystem::path metadata_path = "./bin/" + std::to_string(ORID) + ".meta";
		std::filesystem::path bindata_path = "./bin/" + std::to_string(ORID) + ".bin";

		//gameObjects.clear();
		Core::Buffer metadata;
		Core::syncReadFile(metadata_path, metadata);
		YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
		// check scene name
		if (data["ResourceType"].as<std::string>() != "Mesh") {
			Core::LogManager::Error(std::format("GFX :: Mesh resource not found when deserializing, ORID: {0}", std::to_string(orid)));
			return;
		}
		name = data["Name"].as<std::string>();
		auto vbbl_node = data["VertexBufferLayout"];
		vertexBufferLayout.arrayStride = vbbl_node["ArrayStride"].as<size_t>();
		vertexBufferLayout.stepMode = (RHI::VertexStepMode)vbbl_node["ArrayStride"].as<uint32_t>();
		auto attribute_nodes = vbbl_node["VertexAttributes"];
		for (auto node : attribute_nodes) {
			RHI::VertexAttribute attribute;
			attribute.format = (RHI::VertexFormat)node["VertexFormat"].as<uint32_t>();
			attribute.offset = node["Offset"].as<size_t>();
			attribute.shaderLocation = node["Location"].as<uint32_t>();
			vertexBufferLayout.attributes.push_back(attribute);
		}
		auto ps_node = data["PrimitiveState"];
		primitiveState.topology = (RHI::PrimitiveTopology)ps_node["PrimitiveTopology"].as<uint32_t>();
		primitiveState.stripIndexFormat = (RHI::IndexFormat)ps_node["IndexFormat"].as<uint32_t>();
		primitiveState.frontFace = (RHI::FrontFace)ps_node["FrontFace"].as<uint32_t>();
		primitiveState.cullMode = (RHI::CullMode)ps_node["CullMode"].as<uint32_t>();
		primitiveState.unclippedDepth = ps_node["UnclippedDepth"].as<bool>();
		// load buffers
		size_t vb_size, ib_size, pb_size;
		vb_size = data["VertexBufferSize"].as<size_t>();
		ib_size = data["IndexBufferSize"].as<size_t>();
		pb_size = data["PosOnlyBufferSize"].as<size_t>();
		// load submeshes
		auto submeshes_node = data["Submeshes"];
		for (auto node : submeshes_node) {
			Submesh submesh;
			submesh.baseVertex = node["BaseVertex"].as<uint32_t>();
			submesh.offset = node["Offset"].as<uint32_t>();
			submesh.size = node["Size"].as<uint32_t>();
			submesh.matID = node["MatID"].as<uint32_t>();
			submeshes.push_back(submesh);
		}
		Core::Buffer bindata;
		Core::syncReadFile(bindata_path, bindata);

		vertexBuffer = device->createDeviceLocalBuffer((void*)bindata.data, vb_size,
			(uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::STORAGE);
		indexBuffer = device->createDeviceLocalBuffer((void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size])))[0]), ib_size,
			(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
			(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		if (pb_size != 0) {
			vertexBufferPosOnly = device->createDeviceLocalBuffer((void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size + ib_size])))[0]), pb_size,
				(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
				(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		}
		//Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
		//meshGUIDs.push_back(guid);
		//meshMap[&gltfMesh] = guid;
		//Core::ResourceManager::get()->addResource<GFX::Mesh>(guid, std::move(mesh));
		//Core::ResourceManager::get()->getResource<GFX::Mesh>(guid)->serialize();
		//Core::ResourceManager::get()->database.registerResource(
		//	Core::ResourceManager::get()->getResource<GFX::Mesh>(guid)->ORID, guid);
	}
}