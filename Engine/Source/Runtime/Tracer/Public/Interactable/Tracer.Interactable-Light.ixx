module;
#include <vector>
export module Tracer.Interactable:Light;
import :Interaction;
import :Interaction.SurfaceInteraction;
import :Primitive;
import :Scene;

import Math.Vector;
import Math.Geometry;
import Math.Transform;
import Tracer.Medium;
import Tracer.Spectrum;
import Tracer.Ray;

namespace SIByL::Tracer
{
	export struct VisibilityTester
	{
		VisibilityTester() {}
		VisibilityTester(Interaction const& p0, Interaction const& p1) :_p0(p0), _p1(p1) {}

		auto p0() const noexcept -> Interaction const& { return _p0; }
		auto p1() const noexcept -> Interaction const& { return _p1; }

		auto unoccluded(Scene const& scene) const noexcept -> bool;

		Interaction _p0, _p1;
	};

	export struct Light
	{
		/*
		* Indicates the fundamental light source type.
		*/
		int const flags;


		int const nSamples;

		auto Le(RayDifferential const& ray) const noexcept -> Spectrum { return Spectrum{ 0.f }; }

		virtual auto preprocess(Scene const& scene) noexcept -> void {}

		virtual auto sample_Li(Interaction const& ref, Math::point2 const& u, Math::vec3* wi, float* pdf, VisibilityTester* vis) const noexcept -> Spectrum = 0;
		/**
		* Light's coordinate system with respect to world space.
		* Could implement a light assuming a particular coordinate system,
		* and use transform to place it at arbitrary position & orientations.
		*/
		Math::Transform lightToWorld, worldToLight;
	};
}