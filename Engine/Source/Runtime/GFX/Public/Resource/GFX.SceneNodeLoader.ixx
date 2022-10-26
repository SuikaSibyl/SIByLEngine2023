module;
#include <format>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#pragma warning(disable:4996)
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define JSON_NOEXCEPTION
#include <tinygltf/tiny_gltf.h>
export module GFX.SceneNodeLoader;
import Core.Log;
import Core.ECS;
import Core.Memory;
import Core.Resource.RuntimeManage;
import Math.Vector;
import Math.Matrix;
import Math.Geometry;
import Math.Transform;
import RHI;
import GFX.Resource;
import GFX.GFXManager;
import GFX.Components;

namespace SIByL::GFX
{
    template <typename T>
    struct ArrayAdapter {
        /**
        * Construct an array adapter.
        * @param ptr Pointer to the start of the data, with offset applied
        * @param count Number of elements in the array
        * @param byte_stride Stride betweens elements in the array 
        */ 
        ArrayAdapter(const unsigned char* ptr, size_t count, size_t byte_stride)
            : dataPtr(ptr), elemCount(count), stride(byte_stride) {}

        /// Returns a *copy* of a single element. Can't be used to modify it.
        T operator[](size_t pos) const {
            if (pos >= elemCount)
                throw std::out_of_range(
                    "Tried to access beyond the last element of an array adapter with "
                    "count " +
                    std::to_string(elemCount) + " while getting elemnet number " +
                    std::to_string(pos));
            return *(reinterpret_cast<const T*>(dataPtr + pos * stride));
        }
        /** Pointer to the bytes */
        unsigned const char* dataPtr;
        /** Number of elements in the array */
        const size_t elemCount;
        /** Stride in bytes between two elements */
        const size_t stride;
    };

	inline auto bindModelNodes(GFX::Scene& scene, GameObjectHandle parent, tinygltf::Model const& model, tinygltf::Node const& node, 
        std::unordered_map<tinygltf::Mesh const*, Core::GUID>& meshMap) noexcept -> void;

	inline auto bindMesh(GFX::Scene& scene, GameObjectHandle parent, tinygltf::Model const& model, tinygltf::Node const& node, tinygltf::Mesh const& mesh,
        std::unordered_map<tinygltf::Mesh const*, Core::GUID>& meshMap) noexcept -> void;

	export struct SceneNodeLoader_glTF {
		/** Load glTF file */
		static inline auto loadSceneNode(std::filesystem::path const& path, GFX::Scene& gfxscene) noexcept -> void {
			// use tinygltf to load file
			tinygltf::Model model;
			tinygltf::TinyGLTF loader;
			std::string err;
			std::string warn;
			bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
			if (!warn.empty()) Core::LogManager::Warning(std::format("GFX :: tinygltf :: {0}", warn.c_str()));
			if (!err.empty()) Core::LogManager::Error(std::format("GFX :: tinygltf :: {0}", err.c_str()));
			if (!ret) {
				Core::LogManager::Error("GFX :: tinygltf :: Failed to parse glTF");
				return;
			}
            // Iterate through all the meshes in the glTF file
            // Load meshes into Runtime resource managers.
            RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
            std::vector<Core::GUID> meshGUIDs = {};
            std::unordered_map<tinygltf::Mesh const*, Core::GUID> meshMap = {};
            for (auto const& gltfMesh : model.meshes) {
                std::vector<uint16_t> indexArray_uint16 = {};
                std::vector<float> vertexBuffer = {};
                std::vector<float> vertexBuffer_positionOnly = {};
                std::vector<float> vertexBuffer_normalOnly = {};
                // For each primitive
                for (auto const& meshPrimitive : gltfMesh.primitives) {
                    auto const& indicesAccessor = model.accessors[meshPrimitive.indices];
                    auto const& bufferView = model.bufferViews[indicesAccessor.bufferView];
                    auto const& buffer = model.buffers[bufferView.buffer];
                    auto const dataAddress = buffer.data.data() + bufferView.byteOffset + indicesAccessor.byteOffset;
                    auto const byteStride = indicesAccessor.ByteStride(bufferView);
                    auto const count = indicesAccessor.count;
                    switch (indicesAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_BYTE: {   
                        ArrayAdapter<char> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint16.emplace_back(uint16_t(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                        ArrayAdapter<unsigned char> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint16.emplace_back(uint16_t(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_SHORT: {
                        ArrayAdapter<short> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint16.emplace_back(uint16_t(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        ArrayAdapter<unsigned short> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint16.emplace_back(uint16_t(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_INT: {
                        ArrayAdapter<int> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint16.emplace_back(uint16_t(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                        ArrayAdapter<unsigned int> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint16.emplace_back(uint16_t(originIndexArray[i])); }
                        break;
                    default:
                        break;
                    }
                    float a = 1.f;
                    // We re-arrange the indices so that it describe a simple list of triangles
                    switch (meshPrimitive.mode) {
                    //case TINYGLTF_MODE_TRIANGLE_FAN: // TODO
                    //case TINYGLTF_MODE_TRIANGLE_STRIP: // TODO
                    case TINYGLTF_MODE_TRIANGLES:  // this is the simpliest case to handle
                    {
                        for (auto const& attribute : meshPrimitive.attributes) {
                            auto const attribAccessor = model.accessors[attribute.second];
                            auto const& bufferView = model.bufferViews[attribAccessor.bufferView];
                            auto const& buffer = model.buffers[bufferView.buffer];
                            auto const dataPtr = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
                            auto const byte_stride = attribAccessor.ByteStride(bufferView);
                            auto const count = attribAccessor.count;
                            if (attribute.first == "POSITION") {
                                switch (attribAccessor.type) {
                                case TINYGLTF_TYPE_VEC3: {
                                    switch (attribAccessor.componentType) {
                                    case TINYGLTF_COMPONENT_TYPE_FLOAT:
                                        // 3D vector of float
                                        ArrayAdapter<Math::vec3> positions (dataPtr, count, byte_stride);
                                        for (size_t i{ 0 }; i < indexArray_uint16.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint16[3 * i + 0];
                                            auto f1 = indexArray_uint16[3 * i + 1];
                                            auto f2 = indexArray_uint16[3 * i + 2];
                                            // get the 3 normal vectors for that face
                                            Math::vec3 p0, p1, p2;
                                            p0 = positions[f0];
                                            p1 = positions[f1];
                                            p2 = positions[f2];
                                            // Put them in the array in the correct order
                                            vertexBuffer_positionOnly.push_back(p0.x);
                                            vertexBuffer_positionOnly.push_back(p0.y);
                                            vertexBuffer_positionOnly.push_back(p0.z);

                                            vertexBuffer_positionOnly.push_back(p1.x);
                                            vertexBuffer_positionOnly.push_back(p1.y);
                                            vertexBuffer_positionOnly.push_back(p1.z);

                                            vertexBuffer_positionOnly.push_back(p2.x);
                                            vertexBuffer_positionOnly.push_back(p2.y);
                                            vertexBuffer_positionOnly.push_back(p2.z);
                                        }
                                    }
                                    break;
                                case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                                    switch (attribAccessor.type) {
                                    case TINYGLTF_TYPE_VEC3: {
                                        ArrayAdapter<Math::dvec3> positions(dataPtr, count, byte_stride);
                                        for (size_t i{ 0 }; i < indexArray_uint16.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint16[3 * i + 0];
                                            auto f1 = indexArray_uint16[3 * i + 1];
                                            auto f2 = indexArray_uint16[3 * i + 2];
                                            // get the 3 normal vectors for that face
                                            Math::dvec3 p0, p1, p2;
                                            p0 = positions[f0];
                                            p1 = positions[f1];
                                            p2 = positions[f2];
                                            // Put them in the array in the correct order
                                            vertexBuffer_positionOnly.push_back(p0.x);
                                            vertexBuffer_positionOnly.push_back(p0.y);
                                            vertexBuffer_positionOnly.push_back(p0.z);

                                            vertexBuffer_positionOnly.push_back(p1.x);
                                            vertexBuffer_positionOnly.push_back(p1.y);
                                            vertexBuffer_positionOnly.push_back(p1.z);

                                            vertexBuffer_positionOnly.push_back(p2.x);
                                            vertexBuffer_positionOnly.push_back(p2.y);
                                            vertexBuffer_positionOnly.push_back(p2.z);
                                        }
                                    } break;
                                    default:
                                        // TODO Handle error
                                        break;
                                    }
                                    break;
                                default:
                                    break;
                                }
                                } break;
                                }
                            }
                            if (attribute.first == "NORMAL") {
                                switch (attribAccessor.type) {
                                case TINYGLTF_TYPE_VEC3: {
                                    switch (attribAccessor.componentType) {
                                    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                                        ArrayAdapter<Math::vec3> normals(dataPtr, count, byte_stride);
                                        // For each triangle :
                                        for (size_t i{ 0 }; i < indexArray_uint16.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint16[3 * i + 0];
                                            auto f1 = indexArray_uint16[3 * i + 1];
                                            auto f2 = indexArray_uint16[3 * i + 2];
                                            // get the 3 normal vectors for that face
                                            Math::vec3 n0, n1, n2;
                                            n0 = normals[f0];
                                            n1 = normals[f1];
                                            n2 = normals[f2];
                                            // Put them in the array in the correct order
                                            vertexBuffer_normalOnly.push_back(n0.x);
                                            vertexBuffer_normalOnly.push_back(n0.y);
                                            vertexBuffer_normalOnly.push_back(n0.z);

                                            vertexBuffer_normalOnly.push_back(n1.x);
                                            vertexBuffer_normalOnly.push_back(n1.y);
                                            vertexBuffer_normalOnly.push_back(n1.z);

                                            vertexBuffer_normalOnly.push_back(n2.x);
                                            vertexBuffer_normalOnly.push_back(n2.y);
                                            vertexBuffer_normalOnly.push_back(n2.z);
                                        }
                                    } break;
                                    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                                        ArrayAdapter<Math::dvec3> normals(dataPtr, count, byte_stride);
                                        // IMPORTANT: We need to reorder normals (and texture
                                        // coordinates into "facevarying" order) for each face
                                        // For each triangle :
                                        for (size_t i{ 0 }; i < indexArray_uint16.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint16[3 * i + 0];
                                            auto f1 = indexArray_uint16[3 * i + 1];
                                            auto f2 = indexArray_uint16[3 * i + 2];
                                            // get the 3 normal vectors for that face
                                            Math::dvec3 n0, n1, n2;
                                            n0 = normals[f0];
                                            n1 = normals[f1];
                                            n2 = normals[f2];
                                            // Put them in the array in the correct order
                                            vertexBuffer_normalOnly.push_back(n0.x);
                                            vertexBuffer_normalOnly.push_back(n0.y);
                                            vertexBuffer_normalOnly.push_back(n0.z);

                                            vertexBuffer_normalOnly.push_back(n1.x);
                                            vertexBuffer_normalOnly.push_back(n1.y);
                                            vertexBuffer_normalOnly.push_back(n1.z);

                                            vertexBuffer_normalOnly.push_back(n2.x);
                                            vertexBuffer_normalOnly.push_back(n2.y);
                                            vertexBuffer_normalOnly.push_back(n2.z);
                                        }
                                    } break;
                                    }
                                }
                                }
                            }
                        }
                        break;
                    }
                    default:
                        Core::LogManager::Error("GFX :: tinygltf :: primitive mode not implemented");
                        break;
                    }
                }
                // Assemble vertex buffer
                for (size_t i = 0; i < indexArray_uint16.size(); ++i) {
                    vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
                    vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
                    vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);

                    vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 0]);
                    vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 1]);
                    vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 2]);
                }
                // Create GFX mesh, and add it to resource manager
                GFX::Mesh mesh;
                mesh.vertexBuffer = device->createDeviceLocalBuffer((void*)vertexBuffer.data(), vertexBuffer.size() * sizeof(float), 
                    (uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::STORAGE);
                mesh.indexBuffer = device->createDeviceLocalBuffer((void*)indexArray_uint16.data(), indexArray_uint16.size() * sizeof(uint16_t), 
                    (uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
                    (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
                Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
                meshGUIDs.push_back(guid);
                meshMap[&gltfMesh] = guid;
                Core::ResourceManager::get()->addResource<GFX::Mesh>(guid, std::move(mesh));
            }
			// Bind scene
			GameObjectHandle rootNode = gfxscene.createGameObject(GFX::NULL_GO);
			gfxscene.getGameObject(rootNode)->getEntity().getComponent<TagComponent>()->name = path.filename().string();
			tinygltf::Scene const& scene = model.scenes[model.defaultScene];
			for (size_t i = 0; i < scene.nodes.size(); ++i) {
				assert((scene.nodes[i] >= 0) && (scene.nodes[i] < model.nodes.size()));
				bindModelNodes(gfxscene, rootNode, model, model.nodes[scene.nodes[i]], meshMap);
			}

            GFX::Mesh mesh;
		}
	};

	inline auto bindModelNodes(GFX::Scene& scene, GameObjectHandle parent, tinygltf::Model const& model, tinygltf::Node const& node,
        std::unordered_map<tinygltf::Mesh const*, Core::GUID>& meshMap) noexcept -> void {
		GameObjectHandle sceneNode = scene.createGameObject(parent);
		scene.getGameObject(sceneNode)->getEntity().getComponent<TagComponent>()->name = node.name;
        TransformComponent* transform = scene.getGameObject(sceneNode)->getEntity().getComponent<TransformComponent>();
        Math::mat4 matrix = {
            (float)node.matrix[0],  (float)node.matrix[4],  (float)node.matrix[8],  (float)node.matrix[12],
            (float)node.matrix[1],  (float)node.matrix[5],  (float)node.matrix[9],  (float)node.matrix[13],
            (float)node.matrix[2],  (float)node.matrix[6],  (float)node.matrix[10], (float)node.matrix[14],
            (float)node.matrix[3],  (float)node.matrix[7],  (float)node.matrix[11], (float)node.matrix[15],
        };
        Math::decompose(matrix, &transform->translation, &transform->eulerAngles, &transform->scale);

        // bind mesh
        if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
            bindMesh(scene, sceneNode, model, node, model.meshes[node.mesh], meshMap);
        }
		// bind children nodes
		for (size_t i = 0; i < node.children.size(); ++i) {
			assert((node.children[i] >= 0) && (node.children[i] < model.nodes.size()));
			bindModelNodes(scene, sceneNode, model, model.nodes[node.children[i]], meshMap);
		}
	}

    inline auto bindMesh(GFX::Scene& scene, GameObjectHandle parent, tinygltf::Model const& model, tinygltf::Node const& node, tinygltf::Mesh const& mesh,
        std::unordered_map<tinygltf::Mesh const*, Core::GUID>& meshMap) noexcept -> void {
        scene.getGameObject(parent)->getEntity().addComponent<MeshReference>();
        scene.getGameObject(parent)->getEntity().getComponent<MeshReference>()->mesh =
            Core::ResourceManager::get()->getResource<GFX::Mesh>(meshMap[&mesh]);
    }
}