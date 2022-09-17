module;
#include <string>
export module Tracer.Light:GonioPhotometricLight;
import Math.Vector;
import Math.Geometry;
import Math.Transform;
import Math.Trigonometric;
import Tracer.Interactable;
import Tracer.Medium;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct GonioPhotometricLight :public Light
	{
		//GonioPhotometricLight(Math::Transform const& lightToWorld, MediumInterface const& mediumInterface,
		//	Spectrum const& I, std::string const& texname, float fov)
		//	: Light((int)LightFlags::DeltaPosition, lightToWorld, mediumInterface)
		//	, pLight(lightToWorld* Math::point3(0, 0, 0)), I(I)
		//{
		//	// Create ProjectionLight MIP map
		//	Math::ipoint2 resolution;
		//	// Initialize ProjectionLight projection matrix
		//	// Compute cosine of cone surrounding projection directions
		//}

	private:
		Math::point3 const pLight;
		Spectrum const I;
	};
}