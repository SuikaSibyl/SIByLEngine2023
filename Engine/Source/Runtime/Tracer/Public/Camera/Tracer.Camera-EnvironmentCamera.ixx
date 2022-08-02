module;
export module Tracer.Camera:EnvironmentCamera;
import :Camera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
	export struct EnvironmentCamera :public Camera
	{
	};
}