module;
#include "yaml-cpp/yaml.h"
export module SE.GFX:SerializeUtils;
import SE.Math.Geometric;
using namespace SIByL;

namespace YAML
{
	export template<>
	struct convert<Math::vec3> {
		static NodeAoS encode(Math::vec3 const& rhs) {
			NodeAoS node;
			node.push_back(rhs.x);
			node.push_back(rhs.y);
			node.push_back(rhs.z);
			return node;
		}

		static bool decode(const NodeAoS& node, Math::vec3& rhs) {
			if (!node.IsSequence() || node.size() != 3)
				return false;
			rhs.x = node[0].as<float>();
			rhs.y = node[1].as<float>();
			rhs.z = node[2].as<float>();
			return true;
		}
	};

	export template<>
	struct convert<Math::vec4> {
		static NodeAoS encode(Math::vec4 const& rhs) {
			NodeAoS node;
			node.push_back(rhs.x);
			node.push_back(rhs.y);
			node.push_back(rhs.z);
			node.push_back(rhs.w);
			return node;
		}

		static bool decode(const NodeAoS& node, Math::vec4& rhs) {
			if (!node.IsSequence() || node.size() != 4)
				return false;
			rhs.x = node[0].as<float>();
			rhs.y = node[1].as<float>();
			rhs.z = node[2].as<float>();
			rhs.w = node[3].as<float>();
			return true;
		}
	};

	export inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::vec3 const& v) {
		out << YAML::Flow;
		out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
		return out;
	}

	export inline YAML::Emitter& operator<<(YAML::Emitter& out, Math::vec4 const& v) {
		out << YAML::Flow;
		out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
		return out;
	}
}