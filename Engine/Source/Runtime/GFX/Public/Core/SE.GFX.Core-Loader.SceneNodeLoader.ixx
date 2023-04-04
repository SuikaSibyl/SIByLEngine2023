module;
#include <format>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#pragma warning(disable:4996)
#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define JSON_NOEXCEPTION
#include <tinygltf/tiny_gltf.h>
#include <tiny_obj_loader.h>
export module SE.GFX:SceneNodeLoader;
import :SerializeUtils;
import :GFXConfig;
import :Main;
import :MeshLoader;
import SE.Core.Log;
import SE.Core.ECS;
import SE.Core.Memory;
import SE.Core.Resource;
import SE.Math.Geometric;
import SE.Image;
import SE.RHI;

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

    export struct SceneNodeLoader_obj {
        /** Load obj file */
        static inline auto loadSceneNode(std::filesystem::path const& path, GFX::Scene& gfxscene, MeshLoaderConfig meshConfig = {}) noexcept -> void {
            // load obj file
            std::string inputfile = path.string();
            tinyobj::ObjReaderConfig reader_config;
            reader_config.mtl_search_path = path.parent_path().string(); // Path to material files
            tinyobj::ObjReader reader;
            if (!reader.ParseFromFile(inputfile, reader_config)) {
                if (!reader.Error().empty()) {
                    Core::LogManager::Error("TinyObjReader: " + reader.Error());
                }
                return;
            }
            if (!reader.Warning().empty()) {
                Core::LogManager::Warning("TinyObjReader: " + reader.Warning());
            }
            auto& attrib = reader.GetAttrib();
            auto& shapes = reader.GetShapes();
            auto& materials = reader.GetMaterials();

            uint32_t vertex_offset = 0;
            std::unordered_map<uint64_t, uint32_t> uniqueVertices{};

            std::vector<float>      vertexBufferV = {};
            std::vector<float>      positionBufferV = {};
            std::vector<uint16_t>   indexBufferV = {};
            std::vector<uint32_t>   indexBufferWV = {};

            // check whether tangent is need in mesh attributes
            bool needTangent = false;
            for (auto const& entry : meshConfig.layout.layout)
                if (entry.info == MeshDataLayout::VertexInfo::TANGENT)
                    needTangent = true;

            // Loop over shapes
            GFX::Mesh mesh;
            uint64_t global_index_offset = 0;
            uint32_t submesh_vertex_offset = 0, submesh_index_offset = 0;
            for (size_t s = 0; s < shapes.size(); s++) {
                // Loop over faces(polygon)
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                    size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
                    // require tangent
                    if (fv != 3) {
                        Core::LogManager::Error("GFX :: SceneNodeLoader_obj :: non-triangle geometry not supported when required TANGENT attribute now.");
                        return;
                    }
                    Math::vec3 tangent;
                    Math::vec3 bitangent;
                    if (needTangent) {
                        Math::vec3 positions[3];
                        Math::vec3 normals[3];
                        Math::vec2 uvs[3];
                        for (size_t v = 0; v < fv; v++) {
                            // index finding
                            tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                            positions[v] = {
                                attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                                attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                                attrib.vertices[3 * size_t(idx.vertex_index) + 2] };
                            normals[v] = {
                                attrib.normals[3 * size_t(idx.normal_index) + 0],
                                attrib.normals[3 * size_t(idx.normal_index) + 1],
                                attrib.normals[3 * size_t(idx.normal_index) + 2] };
                            uvs[v] = {
                                attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                                -attrib.texcoords[2 * size_t(idx.texcoord_index) + 1] };
                        }
                        Math::vec3 edge1 = positions[1] - positions[0];
                        Math::vec3 edge2 = positions[2] - positions[0];
                        Math::vec2 deltaUV1 = uvs[1] - uvs[0];
                        Math::vec2 deltaUV2 = uvs[2] - uvs[0];

                        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

                        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
                        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
                        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
                        tangent = Math::normalize(tangent);

                        bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
                        bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
                        bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
                        bitangent = Math::normalize(bitangent);
                    }
                    // Loop over vertices in the face.
                    for (size_t v = 0; v < fv; v++) {
                        // index finding
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        // atrributes filling
                        std::vector<float> vertex = {};
                        std::vector<float> position = {};
                        for (auto const& entry : meshConfig.layout.layout) {
                            // vertex position
                            if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
                                if (entry.format == RHI::VertexFormat::FLOAT32X3) {
                                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                                    vertex.push_back(vx);
                                    vertex.push_back(vy);
                                    vertex.push_back(vz);
                                    if (meshConfig.usePositionBuffer) {
                                        position.push_back(vx);
                                        position.push_back(vy);
                                        position.push_back(vz); 
                                    }
                                }
                                else {
                                    Core::LogManager::Error("GFX :: SceneNodeLoader_obj :: unwanted vertex format for POSITION attributes.");
                                    return;
                                }
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
                                // Check if `normal_index` is zero or positive. negative = no normal data
                                if (idx.normal_index >= 0) {
                                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                                    vertex.push_back(nx);
                                    vertex.push_back(ny);
                                    vertex.push_back(nz);
                                }
                                else {
                                    vertex.push_back(0);
                                    vertex.push_back(0);
                                    vertex.push_back(0);
                                }
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::UV) {
                                if (idx.texcoord_index >= 0) {
                                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                                    vertex.push_back(tx);
                                    vertex.push_back(1 - ty);
                                }
                                else {
                                    vertex.push_back(0);
                                    vertex.push_back(0);
                                }
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::TANGENT) {
                                vertex.push_back(tangent.x);
                                vertex.push_back(tangent.y);
                                vertex.push_back(tangent.z);
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
                                // Optional: vertex colors
                                tinyobj::real_t red = attrib.colors[3 * size_t(idx.vertex_index) + 0];
                                tinyobj::real_t green = attrib.colors[3 * size_t(idx.vertex_index) + 1];
                                tinyobj::real_t blue = attrib.colors[3 * size_t(idx.vertex_index) + 2];
                                vertex.push_back(red);
                                vertex.push_back(green);
                                vertex.push_back(blue);
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {

                            }
                        }

                        if (meshConfig.deduplication) {
                            uint64_t hashed_vertex = hash_vertex(vertex);
                            if (uniqueVertices.count(hashed_vertex) == 0) {
                                uniqueVertices[hashed_vertex] = static_cast<uint32_t>(vertex_offset);
                                vertexBufferV.insert(vertexBufferV.end(), vertex.begin(), vertex.end());
                                positionBufferV.insert(positionBufferV.end(), position.begin(), position.end());
                                ++vertex_offset;
                            }

                            // index filling
                            if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t)
                                indexBufferV.push_back(uniqueVertices[hashed_vertex]);
                            else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T)
                                indexBufferWV.push_back(uniqueVertices[hashed_vertex]);
                        }
                        else {
                            vertexBufferV.insert(vertexBufferV.end(), vertex.begin(), vertex.end());
                            ++vertex_offset;
                            // index filling
                            if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t)
                                indexBufferV.push_back(vertex_offset);
                            else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T)
                                indexBufferWV.push_back(vertex_offset);
                        }
                    }
                    index_offset += fv;
                    // per-face material
                    shapes[s].mesh.material_ids[f];
                }
                global_index_offset += index_offset;

                mesh.submeshes.push_back(GFX::Mesh::Submesh{
                    submesh_index_offset, uint32_t(index_offset),
                    submesh_vertex_offset, uint32_t(0) });
                submesh_index_offset = global_index_offset;
                submesh_vertex_offset = submesh_index_offset;
            }
            // create mesh resource
            {   // register mesh
                mesh.vertexBufferLayout = getVertexBufferLayout(meshConfig.layout);
                if(meshConfig.residentOnHost){
                    mesh.vertexBuffer_host = Core::Buffer(sizeof(float) * vertexBufferV.size());
                    memcpy(mesh.vertexBuffer_host.data, vertexBufferV.data(), mesh.vertexBuffer_host.size);
                    mesh.vertexBufferInfo.onHost = true;
                    mesh.vertexBufferInfo.size = mesh.vertexBuffer_host.size;
                    if (meshConfig.layout.format == RHI::IndexFormat::UINT16_t) {
                        mesh.indexBuffer_host = Core::Buffer(sizeof(uint16_t) * indexBufferV.size());
                        memcpy(mesh.indexBuffer_host.data, indexBufferV.data(), mesh.indexBuffer_host.size);
                        mesh.indexBufferInfo.size = mesh.indexBuffer_host.size;
                        mesh.indexBufferInfo.onHost = true;
                    }
                    else if (meshConfig.layout.format == RHI::IndexFormat::UINT32_T) {
                        mesh.indexBuffer_host = Core::Buffer(sizeof(uint32_t) * indexBufferWV.size());
                        memcpy(mesh.indexBuffer_host.data, indexBufferWV.data(), mesh.indexBuffer_host.size);
                        mesh.indexBufferInfo.size = mesh.indexBuffer_host.size;
                        mesh.indexBufferInfo.onHost = true;
                    }
                    if (meshConfig.usePositionBuffer) {
                        mesh.positionBuffer_host = Core::Buffer(sizeof(float) * positionBufferV.size());
                        memcpy(mesh.positionBuffer_host.data, positionBufferV.data(), mesh.positionBuffer_host.size);
                        mesh.positionBufferInfo.onHost = true;
                        mesh.positionBufferInfo.size = mesh.positionBuffer_host.size;
                    }
                }
            }
            Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
            Core::ORID orid = Core::ResourceManager::get()->database.mapResourcePath(path.string().c_str());
            Core::ResourceManager::get()->database.registerResource(orid, guid);
            Core::ResourceManager::get()->addResource<GFX::Mesh>(guid, std::move(mesh));
            GFX::Mesh* meshResourceRef = Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
            meshResourceRef->ORID = orid;
            meshResourceRef->serialize();
            // bind scene
            GameObjectHandle rootNode = gfxscene.createGameObject(GFX::NULL_GO);
            gfxscene.getGameObject(rootNode)->getEntity().getComponent<TagComponent>()->name = path.filename().string();
            gfxscene.getGameObject(rootNode)->getEntity().addComponent<MeshReference>()->mesh = meshResourceRef;
        }
    };

	export struct SceneNodeLoader_glTF {
		/** Load glTF file */
		static inline auto loadSceneNode(std::filesystem::path const& path, GFX::Scene& gfxscene) noexcept -> void {
#define INDEX_TYPE uint32_t
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
            std::vector<Core::GUID> matGUIDs = {};
            std::unordered_map<tinygltf::Mesh const*, Core::GUID> meshMap = {};
            for (auto const& gltfMesh : model.meshes) {
                std::vector<INDEX_TYPE> indexBuffer_uint = {};
                std::vector<float> vertexBuffer = {};
                std::vector<float> PositionBuffer = {};
                // Create GFX mesh, and add it to resource manager
                GFX::Mesh mesh;
                uint32_t submesh_index_offset = 0;
                uint32_t submesh_vertex_offset = 0;
                // For each primitive
                for (auto const& meshPrimitive : gltfMesh.primitives) {
                    std::vector<INDEX_TYPE> indexArray_uint = {};
                    std::vector<float> vertexBuffer_positionOnly = {};
                    std::vector<float> vertexBuffer_normalOnly = {};
                    std::vector<float> vertexBuffer_uvOnly = {};
                    auto const& indicesAccessor = model.accessors[meshPrimitive.indices];
                    auto const& bufferView = model.bufferViews[indicesAccessor.bufferView];
                    auto const& buffer = model.buffers[bufferView.buffer];
                    auto const dataAddress = buffer.data.data() + bufferView.byteOffset + indicesAccessor.byteOffset;
                    auto const byteStride = indicesAccessor.ByteStride(bufferView);
                    uint64_t const count = indicesAccessor.count;
                    switch (indicesAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_BYTE: {   
                        ArrayAdapter<char> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                        ArrayAdapter<unsigned char> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_SHORT: {
                        ArrayAdapter<short> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        ArrayAdapter<unsigned short> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_INT: {
                        ArrayAdapter<int> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i])); }
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                        ArrayAdapter<unsigned int> originIndexArray(dataAddress, count, byteStride);
                        for (size_t i = 0; i < count; ++i)
                            indexArray_uint.emplace_back(INDEX_TYPE(originIndexArray[i])); }
                        break;
                    default:
                        break;
                    }
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
                                        for (size_t i{ 0 }; i < indexArray_uint.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint[3 * i + 0];
                                            auto f1 = indexArray_uint[3 * i + 1];
                                            auto f2 = indexArray_uint[3 * i + 2];
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
                                        for (size_t i{ 0 }; i < indexArray_uint.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint[3 * i + 0];
                                            auto f1 = indexArray_uint[3 * i + 1];
                                            auto f2 = indexArray_uint[3 * i + 2];
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
                                        for (size_t i{ 0 }; i < indexArray_uint.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint[3 * i + 0];
                                            auto f1 = indexArray_uint[3 * i + 1];
                                            auto f2 = indexArray_uint[3 * i + 2];
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
                                        for (size_t i{ 0 }; i < indexArray_uint.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint[3 * i + 0];
                                            auto f1 = indexArray_uint[3 * i + 1];
                                            auto f2 = indexArray_uint[3 * i + 2];
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
                            if (attribute.first == "TEXCOORD_0") {
                                switch (attribAccessor.type) {
                                case TINYGLTF_TYPE_VEC2: {
                                    switch (attribAccessor.componentType) {
                                    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                                        ArrayAdapter<Math::vec2> uvs(dataPtr, count, byte_stride);
                                        for (size_t i{ 0 }; i < indexArray_uint.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint[3 * i + 0];
                                            auto f1 = indexArray_uint[3 * i + 1];
                                            auto f2 = indexArray_uint[3 * i + 2];
                                            // get the texture coordinates for each triangle's
                                            // vertices
                                            Math::vec2 uv0, uv1, uv2;
                                            uv0 = uvs[f0];
                                            uv1 = uvs[f1];
                                            uv2 = uvs[f2];
                                            // push them in order into the mesh data
                                            vertexBuffer_uvOnly.push_back(uv0.x);
                                            vertexBuffer_uvOnly.push_back(uv0.y);

                                            vertexBuffer_uvOnly.push_back(uv1.x);
                                            vertexBuffer_uvOnly.push_back(uv1.y);

                                            vertexBuffer_uvOnly.push_back(uv2.x);
                                            vertexBuffer_uvOnly.push_back(uv2.y);
                                        }

                                    } break;
                                    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
                                        ArrayAdapter<Math::dvec2> uvs(dataPtr, count, byte_stride);
                                        for (size_t i{ 0 }; i < indexArray_uint.size() / 3; ++i) {
                                            // get the i'th triange's indexes
                                            auto f0 = indexArray_uint[3 * i + 0];
                                            auto f1 = indexArray_uint[3 * i + 1];
                                            auto f2 = indexArray_uint[3 * i + 2];

                                            Math::dvec2 uv0, uv1, uv2;
                                            uv0 = uvs[f0];
                                            uv1 = uvs[f1];
                                            uv2 = uvs[f2];

                                            vertexBuffer_uvOnly.push_back(uv0.x);
                                            vertexBuffer_uvOnly.push_back(uv0.y);

                                            vertexBuffer_uvOnly.push_back(uv1.x);
                                            vertexBuffer_uvOnly.push_back(uv1.y);

                                            vertexBuffer_uvOnly.push_back(uv2.x);
                                            vertexBuffer_uvOnly.push_back(uv2.y);
                                        }
                                    } break;
                                    default:
                                        Core::LogManager::Error("GFX :: tinygltf :: unrecognized vector type for UV");
                                    }
                                } break;
                                default:
                                    Core::LogManager::Error("GFX :: tinygltf :: unreconized componant type for UV");
                                }
                            }
                        }
                        break;
                    }
                    default:
                        Core::LogManager::Error("GFX :: tinygltf :: primitive mode not implemented");
                        break;
                    }
                    // Assemble vertex buffer
                    for (size_t i = 0; i < indexArray_uint.size(); ++i) {
                        vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
                        vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
                        vertexBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);

                        vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 0]);
                        vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 1]);
                        vertexBuffer.push_back(vertexBuffer_normalOnly[i * 3 + 2]);
                        
                        vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 0]);
                        vertexBuffer.push_back(vertexBuffer_uvOnly[i * 2 + 1]);

                        PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 0]);
                        PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 1]);
                        PositionBuffer.push_back(vertexBuffer_positionOnly[i * 3 + 2]);

                        indexBuffer_uint.push_back(i);
                    }
                    mesh.submeshes.push_back(GFX::Mesh::Submesh{
                        submesh_index_offset, uint32_t(indexArray_uint.size()),
                        submesh_vertex_offset, uint32_t(meshPrimitive.material) });
                    submesh_index_offset = indexBuffer_uint.size();
                    submesh_vertex_offset = vertexBuffer.size() / 8;
                }
                mesh.vertexBuffer_device = device->createDeviceLocalBuffer((void*)vertexBuffer.data(), vertexBuffer.size() * sizeof(float),
                    (uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::STORAGE);
                mesh.indexBuffer_device = device->createDeviceLocalBuffer((void*)indexBuffer_uint.data(), indexBuffer_uint.size() * sizeof(INDEX_TYPE),
                    (uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
                    (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
                mesh.positionBuffer_device = device->createDeviceLocalBuffer((void*)PositionBuffer.data(), PositionBuffer.size() * sizeof(float),
                    (uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
                    (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
                mesh.primitiveState.stripIndexFormat = RHI::IndexFormat::UINT32_T;
                Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
                meshGUIDs.push_back(guid);
                meshMap[&gltfMesh] = guid;
                Core::ResourceManager::get()->addResource<GFX::Mesh>(guid, std::move(mesh));
                Core::ResourceManager::get()->getResource<GFX::Mesh>(guid)->serialize();
                Core::ResourceManager::get()->database.registerResource(
                Core::ResourceManager::get()->getResource<GFX::Mesh>(guid)->ORID, guid);
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
        if (node.matrix.size() != 0) {
            Math::mat4 matrix = {
                (float)node.matrix[0],  (float)node.matrix[4],  (float)node.matrix[8],  (float)node.matrix[12],
                (float)node.matrix[1],  (float)node.matrix[5],  (float)node.matrix[9],  (float)node.matrix[13],
                (float)node.matrix[2],  (float)node.matrix[6],  (float)node.matrix[10], (float)node.matrix[14],
                (float)node.matrix[3],  (float)node.matrix[7],  (float)node.matrix[11], (float)node.matrix[15],
            };
            Math::decompose(matrix, &transform->translation, &transform->eulerAngles, &transform->scale);
        }

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