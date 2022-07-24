export module Tracer.Light:Light;
import Math.Transform;
import Tracer.Medium;

namespace SIByL::Tracer
{
	struct Light
	{
		/*
		* Indicates the fundamental light source type.
		*/
		int const flags;


		int const nSamples;


		/**
		* Light's coordinate system with respect to world space.
		* Could implement a light assuming a particular coordinate system,
		* and use transform to place it at arbitrary position & orientations.
		*/
		Math::Transform lightToWorld, worldToLight;
	};
}