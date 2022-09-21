module;
#include <string>
#include <cmath>
#include <memory>
#include <mutex>
#include <Core.h>
module Tracer.Film;
import Core.Memory;
import Math.Vector;
import Math.Geometry;
import Tracer.Filter;
import Tracer.Spectrum;
import Parallelism.Atomic;
import Image.Image;
import Image.Color;

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

	auto Film::getSampleBounds() const noexcept -> Math::ibounds2 {
		Math::bounds2 floatBounds(
			(Math::point2)Math::floor(Math::point2(croppedPixelBounds.pMin) + Math::vec2(0.5f, 0.5f) - Math::vec2(filter->radius)),
			(Math::point2)Math::ceil(Math::point2(croppedPixelBounds.pMax) - Math::vec2(0.5f, 0.5f) + Math::vec2(filter->radius)));
		return Math::ibounds2(floatBounds);
	}

	auto Film::getFilmTile(Math::ibounds2 const& sampleBounds) noexcept -> Scope<FilmTile> {
		// bound image pixels that samples in sampleBounds contribute to
		Math::vec2 halfPixel = Math::vec2{ 0.5f,0.5f };
		Math::bounds2 floatBounds = (Math::bounds2)sampleBounds;
		Math::ipoint2 p0 = (Math::ipoint2)Math::ceil(floatBounds.pMin - halfPixel - filter->radius);
		Math::ipoint2 p1 = (Math::ipoint2)Math::floor(floatBounds.pMax - halfPixel + filter->radius) + Math::ipoint2{ 1,1 };
		Math::ibounds2 tilePixelBounds = Math::intersect(Math::ibounds2{ p0,p1 }, croppedPixelBounds);
		return std::unique_ptr<FilmTile>(new FilmTile(tilePixelBounds, filter->radius, filterTable, filterTableWidth));
	}

	auto Film::mergeFilmTile(Scope<FilmTile> tile) noexcept -> void {
		std::lock_guard<std::mutex> lock(mutex);
		for (Math::ipoint2 pixel : tile->getPixelBounds()) {
			// merge pixel into Film::pixels
			FilmTilePixel const& tilePixel = tile->getPixel(pixel);
			Pixel& mergePixel = getPixel(pixel);
			float xyz[3];
			tilePixel.contribSum.toXYZ(xyz);
			for (int i = 0; i < 3; ++i)
				mergePixel.xyz[i] += xyz[i];
			mergePixel.filterWeightSum += tilePixel.filterWeightSum;
		}
	}

	auto Film::getPixel(Math::ipoint2 const& p) noexcept -> Pixel& {
		int width = croppedPixelBounds.pMax.x - croppedPixelBounds.pMin.x;
		int offset = (p.x - croppedPixelBounds.pMin.x) +
			(p.y - croppedPixelBounds.pMin.y) * width;
		return pixels[offset];
	}

	auto Film::setImage(Spectrum const* img) const noexcept -> void {
		int nPixels = croppedPixelBounds.surfaceArea();
		for (int i = 0; i < nPixels; ++i) {
			Pixel& p = pixels[i];
			img[i].toXYZ(p.xyz);
			p.filterWeightSum = 1;
			p.splatXYZ[0] = p.splatXYZ[1] = p.splatXYZ[2] = 0;
		}
	}

	auto Film::addSplat(Math::point2 const& p, Spectrum const& v) noexcept -> void {
		if (Math::insideExclusive((Math::ipoint2)p, croppedPixelBounds))
			return;
		float xyz[3];
		v.toXYZ(xyz);
		Pixel& pixel = getPixel((Math::ipoint2)p);
		for (int i = 0; i < 3; ++i)
			pixel.splatXYZ[i].add(xyz[i]);
	}

	auto Film::writeImage(Image::Image<Image::COLOR_R8G8B8_UINT>& image, float splatScale) noexcept -> void {
		// convert image to RGBand compute final pixel values
		std::unique_ptr<float[]> rgb(new float[3 * croppedPixelBounds.surfaceArea()]);
		int offset = 0;
		for (Math::ipoint2 p : croppedPixelBounds) {
			// Convert pixel XYZ color to RGB
			Pixel& pixel = getPixel(p);
			XYZToRGB(pixel.xyz, &rgb[3 * offset]);
			// Normalize pixel with weight sum 
			float const filterWeightSum = pixel.filterWeightSum;
			if (filterWeightSum != 0) {
				float const invWt = 1.f / filterWeightSum;
				rgb[3 * offset + 0] = std::max(0.f, rgb[3 * offset + 0] * invWt);
				rgb[3 * offset + 1] = std::max(0.f, rgb[3 * offset + 1] * invWt);
				rgb[3 * offset + 2] = std::max(0.f, rgb[3 * offset + 2] * invWt);
			}
			// Add splat value at pixel
			float splatRGB[3];
			float splatXYZ[3] = { pixel.splatXYZ[0],pixel.splatXYZ[1], pixel.splatXYZ[2] };
			XYZToRGB(splatXYZ, splatRGB);
			rgb[3 * offset + 0] += splatScale * splatRGB[0];
			rgb[3 * offset + 1] += splatScale * splatRGB[1];
			rgb[3 * offset + 2] += splatScale * splatRGB[2];
			// Scale pixel value by scale
			rgb[3 * offset + 0] *= scale;
			rgb[3 * offset + 1] *= scale;
			rgb[3 * offset + 2] *= scale;
			++offset;
		}
		// write RGB image
		offset = 0;
		for (Math::ipoint2 p : croppedPixelBounds) {
			image[p.y][p.x] = Image::COLOR_R8G8B8_UINT{ 
				(uint8_t)(255 * rgb[3 * offset + 0]),
				(uint8_t)(255 * rgb[3 * offset + 1]),
				(uint8_t)(255 * rgb[3 * offset + 2])};
			++offset;
		}
	}

	FilmTile::FilmTile(Math::ibounds2 const& pixelBounds, Math::vec2 const& filterRadius, float const* filterTable, int filterTableSize)
		: pixelBounds(pixelBounds), filterRadius(filterRadius)
		, invFilterRadius(1.f / filterRadius.x, 1.f / filterRadius.y)
		, filterTable(filterTable), filterTableSize(filterTableSize)
	{
		pixels = std::vector<FilmTilePixel>(std::max(0, pixelBounds.surfaceArea()));
	}

	auto FilmTile::addSample(Math::point2 const& pFilm, Spectrum const& L, float sampleWeight) noexcept -> void {
		// compute sample's raster bounds
		Math::point2 pFilmDiscrete = pFilm - Math::vec2{ 0.5f,0.5f };
		Math::ipoint2 p0 = (Math::ipoint2)Math::ceil(pFilmDiscrete - filterRadius);
		Math::ipoint2 p1 = (Math::ipoint2)Math::floor(pFilmDiscrete + filterRadius) + Math::ipoint2{ 1,1 };
		p0 = Math::max(p0, pixelBounds.pMin);
		p1 = Math::min(p1, pixelBounds.pMax);
		// loop over filter support and add sample to pixel arrays
		//  precompute x and y filter table offsets
		int* ifx = (int*)Alloca(int, size_t(p1.x - p0.x));
		for (int x = p0.x; x < p1.x; ++x) {
			float fx = std::abs((x - pFilmDiscrete.x) * invFilterRadius.x * filterTableSize);
			ifx[x - p0.x] = std::min((int)std::floor(fx), filterTableSize - 1);
		}
		int* ify = (int*)Alloca(int, size_t(p1.y - p0.y));
		for (int y = p0.y; y < p1.y; ++y) {
			float fy = std::abs((y - pFilmDiscrete.y) * invFilterRadius.y * filterTableSize);
			ify[y - p0.y] = std::min((int)std::floor(fy), filterTableSize - 1);
		}
		//  loop over filter support
		for (int y = p0.y; y < p1.y; ++y)
			for (int x = p0.x; x < p1.x; ++x) {
				// evaluate filter values at (x,y) pixel
				int offset = ify[y - p0.y] * filterTableSize + ifx[x - p0.x];
				float filterWeight = filterTable[offset];
				// update pixel values with filtered sample contribution
				FilmTilePixel& pixel = getPixel(Math::ipoint2(x, y));
				pixel.contribSum += L * sampleWeight * filterWeight;
				pixel.filterWeightSum += filterWeight;
			}
	}

	auto FilmTile::getPixel(Math::ipoint2 const& p) noexcept -> FilmTilePixel& {
		int width = pixelBounds.pMax.x - pixelBounds.pMin.x;
		int offset = (p.x - pixelBounds.pMin.x) +
			(p.y - pixelBounds.pMin.y) * width;
		return pixels[offset];
	}

	auto FilmTile::getPixelBounds() const noexcept -> Math::ibounds2 {
		return pixelBounds;
	}
}
