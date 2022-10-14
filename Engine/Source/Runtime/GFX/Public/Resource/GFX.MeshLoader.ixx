module;
#include <vector>
#include <string>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <filesystem>
export module GFX.MeshLoader;
import Core.Log;
import Core.Memory;
import RHI;

namespace SIByL::GFX
{
    /** Mesh data layout */
    export struct MeshDataLayout {
        /** info types in vertex */
        enum struct VertexInfo {
            POSITION,
            NORMAL,
            UV,
            COLOR,
            CUSTOM,
        };
        /** an entry of the layout */
        struct Entry {
            RHI::VertexFormat  format;
            VertexInfo         info;
        };
        /* the list of vertex layout */
        std::vector<Entry> layout;
        /* index format */
        RHI::IndexFormat format;
    };

	export struct MeshLoader_OBJ {
        static auto loadOBJ(std::filesystem::path const& path, 
            MeshDataLayout const& layout,
            Core::Buffer* vertexBuffer,
            Core::Buffer* indexBuffer) noexcept -> void {
            /** buffers */
            std::vector<float>      vertexBufferV = {};
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
                        // index filling
                        if (layout.format == RHI::IndexFormat::UINT16_t)
                            indexBufferV.push_back(index_offset + v);
                        else if (layout.format == RHI::IndexFormat::UINT32_T)
                            indexBufferWV.push_back(index_offset + v);

                        for (auto const& entry : layout.layout) {
                            // vertex position
                            if (entry.info == MeshDataLayout::VertexInfo::POSITION) {
                                if (entry.format == RHI::VertexFormat::FLOAT32X3) {
                                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                                    vertexBufferV.push_back(vx);
                                    vertexBufferV.push_back(vy);
                                    vertexBufferV.push_back(vz);
                                }
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::NORMAL) {
                                // Check if `normal_index` is zero or positive. negative = no normal data
                                if (idx.normal_index >= 0) {
                                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                                    vertexBufferV.push_back(nx);
                                    vertexBufferV.push_back(ny);
                                    vertexBufferV.push_back(nz);
                                }
                                else {
                                    vertexBufferV.push_back(0);
                                    vertexBufferV.push_back(0);
                                    vertexBufferV.push_back(0);
                                }
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::UV) {
                                if (idx.texcoord_index >= 0) {
                                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                                    vertexBufferV.push_back(tx);
                                    vertexBufferV.push_back(ty);
                                }
                                else {
                                    vertexBufferV.push_back(0);
                                    vertexBufferV.push_back(0);
                                }
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::COLOR) {
                                // Optional: vertex colors
                                 tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                                 tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                                 tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
                                 vertexBufferV.push_back(red);
                                 vertexBufferV.push_back(green);
                                 vertexBufferV.push_back(blue);
                            }
                            else if (entry.info == MeshDataLayout::VertexInfo::CUSTOM) {

                            }
                        }
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
		}
	};
}