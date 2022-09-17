module;
export module Tracer.Camera:ProjectiveCamera;
import Tracer.Medium;
import Tracer.Film;
import Tracer.Ray;
import Math.Geometry;
import Math.Transform;

namespace SIByL::Tracer
{
	export struct ProjectiveCamera :public Camera
	{
		ProjectiveCamera(
			Math::AnimatedTransform const& cameraToWorld, 
			Math::Transform const& cameraToScreen,
			Math::bounds2 const& screenWindow,
			float shutterOpen, float shutterClose, 
			float lensr, float focald,
			Film* film, Medium const* medium);

		/**
		* Three different coordinates
		* Screen space: defined on the film plane
		*				@z: range in [0,1], corresponding to near/far clip plane
		* NDC space:	@x/y: space ranges from (0,0) to (1,1), with (0,0) being the upper-left corner
		*				@z: Depth values are the same as in screen space
		* Raster space: Almost the same as NDC space, except x&y range
		*				@x/y: space ranges from (0,0) to (resolution.xy)
		*				@z: Depth values are the same as in screen space
		*/
		Math::Transform screenToRaster;
		Math::Transform rasterToScreen;
		Math::Transform cameraToScreen;
		Math::Transform rasterToCamera;
	};
}