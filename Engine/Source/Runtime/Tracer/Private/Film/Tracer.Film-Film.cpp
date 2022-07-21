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
		// Allocate film image storage
		pixels = std::unique_ptr<Pixel[]>(new Pixel[croppedPixelBounds.surfaceArea()]);
		// Precompute filter weight table
		int offset = 0;
		for (int y = 0; y < filterTableWidth; ++y) {
			for (int x = 0; x < filterTableWidth; ++x, ++offset) {
				Math::point2 p;
				p.x = (x + 0.5f) * filter->radius.x / filterTableWidth;
				p.y = (y + 0.5f) * filter->radius.y / filterTableWidth;
				filterTable[offset] = filter->evaluate(p);
			}
		}
	}

	auto Film::getPhysicalExtent() const noexcept -> Math::bounds2 {
		float aspect = (float)fullResolution.y / (float)fullResolution.x;
		float x = std::sqrt(diagonal * diagonal / (1 + aspect * aspect));
		float y = aspect * x;
		return Math::bounds2(Math::point2(-x / 2, -y / 2), Math::point2(x / 2, y / 2));
	}
}
