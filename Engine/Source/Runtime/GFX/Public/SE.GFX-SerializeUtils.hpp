#pragma once
#include "yaml-cpp/yaml.h"
#include <SE.Math.Geometric.hpp>
using namespace SIByL;

namespace YAML {
SE_EXPORT template <>
struct convert<Math::vec2> {
  static NodeAoS encode(Math::vec2 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y); return node; }
  static bool decode(const NodeAoS& node, Math::vec2& rhs) {
    if (!node.IsSequence() || node.size() != 2) return false;
    rhs.x = node[0].as<float>(); rhs.y = node[1].as<float>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::ivec2> {
  static NodeAoS encode(Math::ivec2 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y); return node; }
  static bool decode(const NodeAoS& node, Math::ivec2& rhs) {
    if (!node.IsSequence() || node.size() != 2) return false;
    rhs.x = node[0].as<int>(); rhs.y = node[1].as<int>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::uvec2> {
  static NodeAoS encode(Math::uvec2 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y); return node; }
  static bool decode(const NodeAoS& node, Math::uvec2& rhs) {
    if (!node.IsSequence() || node.size() != 2) return false;
    rhs.x = node[0].as<uint32_t>(); rhs.y = node[1].as<uint32_t>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::svec2> {
  static NodeAoS encode(Math::svec2 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y); return node; }
  static bool decode(const NodeAoS& node, Math::svec2& rhs) {
    if (!node.IsSequence() || node.size() != 2) return false;
    rhs.x = node[0].as<size_t>(); rhs.y = node[1].as<size_t>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::vec3> {
  static NodeAoS encode(Math::vec3 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y);
    node.push_back(rhs.z); return node; }
  static bool decode(const NodeAoS& node, Math::vec3& rhs) {
    if (!node.IsSequence() || node.size() != 3) return false;
    rhs.x = node[0].as<float>(); rhs.y = node[1].as<float>();
    rhs.z = node[2].as<float>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::ivec3> {
  static NodeAoS encode(Math::ivec3 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y);
    node.push_back(rhs.z); return node; }
  static bool decode(const NodeAoS& node, Math::ivec3& rhs) {
    if (!node.IsSequence() || node.size() != 3) return false;
    rhs.x = node[0].as<int>(); rhs.y = node[1].as<int>();
    rhs.z = node[2].as<int>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::uvec3> {
  static NodeAoS encode(Math::uvec3 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y);
    node.push_back(rhs.z); return node; }
  static bool decode(const NodeAoS& node, Math::uvec3& rhs) {
    if (!node.IsSequence() || node.size() != 3) return false;
    rhs.x = node[0].as<uint32_t>(); rhs.y = node[1].as<uint32_t>();
    rhs.z = node[2].as<uint32_t>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::vec4> {
  static NodeAoS encode(Math::vec4 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y);
    node.push_back(rhs.z); node.push_back(rhs.w); return node; }
  static bool decode(const NodeAoS& node, Math::vec4& rhs) {
    if (!node.IsSequence() || node.size() != 4) return false;
    rhs.x = node[0].as<float>(); rhs.y = node[1].as<float>();
    rhs.z = node[2].as<float>(); rhs.w = node[3].as<float>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::ivec4> {
  static NodeAoS encode(Math::ivec4 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y);
    node.push_back(rhs.z); node.push_back(rhs.w); return node; }
  static bool decode(const NodeAoS& node, Math::ivec4& rhs) {
    if (!node.IsSequence() || node.size() != 4) return false;
    rhs.x = node[0].as<int>(); rhs.y = node[1].as<int>();
    rhs.z = node[2].as<int>(); rhs.w = node[3].as<int>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::uvec4> {
  static NodeAoS encode(Math::uvec4 const& rhs) {
    NodeAoS node; node.push_back(rhs.x); node.push_back(rhs.y);
    node.push_back(rhs.z); node.push_back(rhs.w); return node; }
  static bool decode(const NodeAoS& node, Math::uvec4& rhs) {
    if (!node.IsSequence() || node.size() != 4) return false;
    rhs.x = node[0].as<uint32_t>(); rhs.y = node[1].as<uint32_t>();
    rhs.z = node[2].as<uint32_t>(); rhs.w = node[3].as<uint32_t>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::mat3> {
  static NodeAoS encode(Math::mat3 const& rhs) {
    NodeAoS node; for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) node.push_back(rhs.data[i][j]); return node; }
  static bool decode(const NodeAoS& node, Math::mat3& rhs) {
    if (!node.IsSequence() || node.size() != 9) return false;
    int index = 0; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) 
        rhs.data[i][j] = node[index++].as<float>(); return true; }
};

SE_EXPORT template <>
struct convert<Math::mat4> {
  static NodeAoS encode(Math::mat4 const& rhs) {
    NodeAoS node; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) 
        node.push_back(rhs.data[i][j]); return node; }
  static bool decode(const NodeAoS& node, Math::mat4& rhs) {
    if (!node.IsSequence() || node.size() != 16) return false;
    int index = 0; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) 
        rhs.data[i][j] = node[index++].as<float>(); return true; }
};

SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::vec2 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << YAML::EndSeq; return out; }
SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::ivec2 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << YAML::EndSeq; return out; }
SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::uvec2 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << YAML::EndSeq; return out; }
SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::svec2 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << YAML::EndSeq; return out; }

SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::vec3 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq; return out; }
SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::ivec3 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq; return out; }
SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::uvec3 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq; return out; }

SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::vec4 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq; return out; }
SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::ivec4 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq; return out; }
SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::uvec4 const& v) {
  out << YAML::Flow << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq; return out; }

SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::mat3 const& m) {
  out << YAML::Flow << YAML::BeginSeq; for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out << m.data[i][j]; out << YAML::EndSeq; return out; }

SE_EXPORT inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::mat4 const& m) {
  out << YAML::Flow << YAML::BeginSeq; for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) out << m.data[i][j]; out << YAML::EndSeq; return out; }
}  // namespace YAML