module;
#include <string>
export module Tracer.Light:ProjectionLight;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Interactable;
import Tracer.Medium;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct ProjectionLight :public Light
	{
		ProjectionLight(Math::Transform const& lightToWorld, MediumInterface const& mediumInterface,
			Spectrum const& I, std::string const& texname, float fov)
			: Light((int)LightFlags::DeltaPosition, lightToWorld, mediumInterface)
			, pLight(lightToWorld* Math::point3(0, 0, 0)), I(I) 
		{
			// Create ProjectionLight MIP map
			Math::ipoint2 resolution;
			// Initialize ProjectionLight projection matrix
			// Compute cosine of cone surrounding projection directions
		}

	private:
		Math::point3 const pLight;
		Spectrum const I;
	};
}