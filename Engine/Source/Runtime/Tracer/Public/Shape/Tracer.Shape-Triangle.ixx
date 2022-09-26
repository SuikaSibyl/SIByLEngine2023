module;
#include <vector>
#include <memory>
export module Tracer.Shape:Triangle;
import Math.Transform;
import Math.Vector;
import Math.Geometry;
import Math.Common;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	export struct TriangleMesh {
		TriangleMesh(Math::Transform const& objectToWorld,
			int nTriangles, int const* vertexIndices, int nVertices,
			Math::point3 const* P, Math::vec3 const* S, Math::normal3 const* N);

		int const nTriangles, nVertices;
		std::vector<int> vertexIndices;
		std::unique_ptr<Math::point3[]>  p;
		std::unique_ptr<Math::normal3[]> n;
		std::unique_ptr<Math::vec3[]>	 s;
		std::unique_ptr<Math::point2[]>	 uv;
		Texture<float>* alphaMask = nullptr;
	};

	export struct Triangle :public Shape
	{
	};
}