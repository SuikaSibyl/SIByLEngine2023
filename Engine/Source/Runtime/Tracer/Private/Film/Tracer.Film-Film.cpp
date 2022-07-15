module;
#include <string>
#include <cmath>
#include <memory>
module Tracer.Film:Film;
import Tracer.Film;
import Math.Vector;
import Math.Geometry;
import Tracer.Filter;

namespace SIByL::Tracer
{
	Film::Film(Math::ipoint2 const& resolution, Math::bounds2 const& cropWindow,
		std::unique_ptr<Filter> filt, float diagonal,
		std::string const& filename, float scale)
		: fullResolution(resolution)
		, diagonal(diagonal)
		, filter(std::move(filt))
		, filename(filename)
		, scale(scale)
	{
		resolution.x;
		// Compute film image bounds
		croppedPixelBounds =
			Math::ibounds2(Math::ipoint2(std::ceil(fullResolution.x * cropWindow.pMin.x),
										 std::ceil(fullResolution.y * cropWindow.pMin.y)),
						   Math::ipoint2(std::ceil(fullResolution.x * cropWindow.pMax.x),
										 std::ceil(fullResolution.y * cropWindow.pMax.y)));
	}
}
