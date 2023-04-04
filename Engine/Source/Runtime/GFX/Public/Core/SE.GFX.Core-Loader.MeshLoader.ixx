module;
#include <array>
#include <vector>
#include <string>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
export module SE.GFX:MeshLoader;
import :SerializeUtils;
import :GFXConfig;
import :Main;
import SE.Core.Log;
import SE.Core.Memory;
import SE.Core.Resource;
import SE.RHI;

namespace SIByL::GFX
{
    static std::array<uint64_t, 24> primes = { 3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97 };

    export inline auto hash_vertex(std::vector<float> const& v) noexcept -> std::uint64_t {
        std::uint64_t hash = reinterpret_cast<std::uint32_t const&>(v[0]);
        for (size_t n = 1; n < std::min(primes.size(), v.size()); ++n) {
            std::uint64_t floated = reinterpret_cast<std::uint32_t const&>(v[n]);
            hash += primes[n] * floated;
        }
        return hash;
    }
    
    export inline auto getVertexBufferLayout(MeshDataLayout const& mdl) noexcept -> RHI::VertexBufferLayout {
        RHI::VertexBufferLayout vbl;
        uint32_t offset = 0;
        for (size_t i = 0; i < mdl.layout.size(); ++i) {
            GFX::MeshDataLayout::Entry const& entry = mdl.layout[i];
            RHI::VertexAttribute attr;
            attr.format = entry.format;
            attr.offset = offset;
            attr.shaderLocation = i;
            vbl.attributes.push_back(attr);
            offset += getVertexFormatSize(entry.format);
        }
        vbl.arrayStride = offset;
        return vbl;
    }

	export struct MeshLoader_OBJ {
        static auto loadOBJ(std::filesystem::path const& path, 
            MeshDataLayout const& layout,
            Core::Buffer* vertexBuffer,
            Core::Buffer* indexBuffer,
            Core::Buffer* vertexPosOnlyBuffer = nullptr) noexcept -> void {
            /** buffers */
            std::vector<float>      vertexBufferV = {};
            std::vector<float>      vertexBufferPosOnlyV = {};
            std::vector<uint16_t>   indexBufferV = {};
            std::vector<uint32_t>   indexBufferWV = {};

            std::string inputfile = path.string();
            tinyobj::ObjReaderConfig reader_config;
            reader_config.mtl_search_path = path.parent_path().string(); // Path to material files

            tinyobj::ObjReader reader;
            if (!reader.ParseFromFile(inputfile, reader_config)) {
                if (!reader.Error().empty()) {
                    Core::LogManager::Error("TinyObjReader: " + reader.Error());
                }
                exit(1);
            }

            if (!reader.Warning().empty()) {
                Core::LogManager::Warning("TinyObjReader: " + reader.Warning());
            }

            auto& attrib = reader.GetAttrib();
            auto& shapes = reader.GetShapes();
            auto& materials = reader.GetMaterials();

            uint32_t vertex_offset = 0;
            std::unordered_map<uint64_t, uint32_t> uniqueVertices{};

            // Loop over shapes
            for (size_t s = 0; s < shapes.size(); s++) {
                // Loop over faces(polygon)
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                    size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                    // Loop over vertices in the face.
                    for (size_t v = 0; v < fv; v++) {
                        // index finding
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        // vertex finding
                        std::vector<float> vertex = {};
                        std::vector<float> position = {};
                        for (auto const& entry : layout.layout) {
                            // vertex position
                            if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
                                if (entry.format == RHI::VertexFormat::FLOAT32X3) {
                                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                                    vertex.push_back(vx);
                                    vertex.push_back(vy);
                                    vertex.push_back(vz);
                                    if (vertexPosOnlyBuffer) {
                                        position.push_back(vx);
                                        position.push_back(vy);
                                        position.push_back(vz);
                                    }
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
                                    vertex.push_back(ty);
                                }
                                else {
                                    vertex.push_back(0);
                                    vertex.push_back(0);
                                }
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
                                // Optional: vertex colors
                                 tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                                 tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                                 tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
                                 vertex.push_back(red);
                                 vertex.push_back(green);
                                 vertex.push_back(blue);
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {

                            }
                        }

                        uint64_t hashed_vertex = hash_vertex(vertex);
                        if (uniqueVertices.count(hashed_vertex) == 0) {
                            uniqueVertices[hashed_vertex] = static_cast<uint32_t>(vertex_offset);
                            vertexBufferV.insert(vertexBufferV.end(), vertex.begin(), vertex.end());
                            vertexBufferPosOnlyV.insert(vertexBufferPosOnlyV.end(), position.begin(), position.end());
                            ++vertex_offset;
                        }
                        else {
                            float a = 1.f;
                        }

                        // index filling
                        if (layout.format == RHI::IndexFormat::UINT16_t)
                            indexBufferV.push_back(uniqueVertices[hashed_vertex]);
                        else if (layout.format == RHI::IndexFormat::UINT32_T)
                            indexBufferWV.push_back(uniqueVertices[hashed_vertex]);
                    }

                    index_offset += fv;

                    // per-face material
                    shapes[s].mesh.material_ids[f];
                }
            }

            if (vertexBuffer) {
                *vertexBuffer = Core::Buffer(sizeof(float) * vertexBufferV.size());
                memcpy(vertexBuffer->data, vertexBufferV.data(), vertexBuffer->size);
            }
            if (indexBuffer) {
                *indexBuffer = Core::Buffer(sizeof(uint16_t) * indexBufferV.size());
                memcpy(indexBuffer->data, indexBufferV.data(), indexBuffer->size);
            }
            if (vertexPosOnlyBuffer) {
                *vertexPosOnlyBuffer = Core::Buffer(sizeof(float) * vertexBufferPosOnlyV.size());
                memcpy(vertexPosOnlyBuffer->data, vertexBufferPosOnlyV.data(), vertexPosOnlyBuffer->size);
            }
		}

        static auto loadMeshResource(std::filesystem::path const& path,
            MeshDataLayout const& layout,
            bool usePosOnlyBuffer) noexcept -> Core::GUID {
            Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
            Core::Buffer vertex;
            Core::Buffer index;
            Core::Buffer vertexPosOnly;
            GFX::MeshLoader_OBJ::loadOBJ(path, layout, &vertex, &index, &vertexPosOnly);
            GFX::Mesh mesh;
            mesh.vertexBuffer_device = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
                (void*)vertex.data, vertex.size,
                (uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::STORAGE);
            mesh.indexBuffer_device = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
                (void*)index.data, index.size,
                (uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
                (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
            mesh.positionBuffer_device = GFX::GFXManager::get()->rhiLayer->getDevice()->createDeviceLocalBuffer(
                (void*)vertexPosOnly.data, vertexPosOnly.size,
                (uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
                (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
            Core::ResourceManager::get()->addResource<GFX::Mesh>(guid, std::move(mesh));
            return guid;
        }
	};
}