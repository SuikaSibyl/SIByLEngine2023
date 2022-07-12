export module GFX.Camera:Camera;
import GFX.Medium;
import Math.Geometry;

namespace SIByL::GFX
{
	export struct CameraSample {
		Math::point2 pFilm;
		Math::point2 pLens;
		float time;
	};

	export struct Camera
	{
		auto generateRay(CameraSample const& sample, Math::ray3* ray) noexcept -> void;
		auto generateRayDifferential() noexcept -> void;


		float const shutterOpen, shutterClose;
		Medium* medium;
	};
}