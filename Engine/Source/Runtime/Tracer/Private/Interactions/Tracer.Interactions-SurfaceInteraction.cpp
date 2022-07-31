module Tracer.Interactions:SurfaceInteraction;
import Tracer.Interactions;
import :Interaction;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	SurfaceInteraction::SurfaceInteraction()
		: Interaction(Math::vec3{}, Math::normal3{}, Math::vec3{}, Math::vec3{}, 0, nullptr)
	{}

	SurfaceInteraction::SurfaceInteraction(Math::point3 const& p, Math::point3 const& pError,
		Math::point2 const& uv, Math::vec3 const& wo,
		Math::vec3 const& dpdu, Math::vec3 const& dpdv,
		Math::normal3 const& dndu, Math::vec3 const& dndv,
		float time, Shape const* shape)
		: Interaction(p, Math::normal3(Math::normalize(Math::cross(dpdu, dpdv))), pError, wo, time, nullptr)
	{}
}