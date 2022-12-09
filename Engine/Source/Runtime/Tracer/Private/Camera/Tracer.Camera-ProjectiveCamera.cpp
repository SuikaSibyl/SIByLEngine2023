module;
module Tracer.Camera:ProjectiveCamera;
import Tracer.Camera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import SE.Math.Geometric;

namespace SIByL::Tracer
{
	ProjectiveCamera::ProjectiveCamera(
		Math::AnimatedTransform const& cameraToWorld,
		Math::Transform const& cameraToScreen,
		Math::bounds2 const& screenWindow,
		float shutterOpen, float shutterClose,
		float lensr, float focald,
		Film* film, Medium const* medium)
		: Camera(cameraToWorld, shutterOpen, shutterClose, film, medium)
		, cameraToScreen(cameraToScreen)
	{
		// Compute projective camera screen transformations
		// Note, y-axis is inversed in screen-to-raster process
		screenToRaster = Math::scale((float)film->fullResolution.x, (float)film->fullResolution.y, 1.f)
			* Math::scale(1.f / (screenWindow.pMax.x - screenWindow.pMin.x), 1.f / (screenWindow.pMin.y - screenWindow.pMax.y), 1.f)
			* Math::translate(Math::vec3(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));
		rasterToScreen = Math::inverse(screenToRaster);

		rasterToCamera = Math::inverse(cameraToScreen) * rasterToScreen;
	}


}