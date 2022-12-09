module;
#include <string>
#include <cmath>
#include <vector>
#include <memory>
#include <mutex>
export module Tracer.Film;
import SE.Core.Memory;
import SE.Math.Geometric;
import Tracer.Filter;
import Tracer.Spectrum;
import SE.Parallelism;
import SE.Image;

namespace SIByL::Tracer
{
	struct FilmTile;
	/*
	* Models the sensing device in the simulated camera.
	* Determine the each camera ray sample's contribution to the pixels around the point
	* on the film plane and updates its representation of the image.
	*/
	export struct Film
	{
		/**
		* @param resolution: Resolution of the image in pixels
		* @param cropWindow: Specify a subset of image to render, In NDC space with each coordinate ranging in [0-1]
		*					 Film only allocates space for and store pixel values in the region inside the crop window.
		* @param diagonal  : Length of the diagnoal of the film's physical area in millimeters
		* @param filt	   : A filter function
		* @param filename  : Filename for the output image
		*/
		Film(Math::ipoint2 const& resolution, Math::bounds2 const& cropWindow,
			std::unique_ptr<Filter> filt, float diagonal,
			std::string const& filename, float scale);

		/** Overall resolution of the image in pixels */
		Math::ipoint2 const fullResolution;
		/** Pixel bounds from the upper-left  to the lower-right corners of the crop window */
		Math::ibounds2 croppedPixelBounds;

		/** Get actual extent of the film in the scene */
		auto getPhysicalExtent() const noexcept -> Math::bounds2;

		auto getSampleBounds() const noexcept -> Math::ibounds2;

		auto getFilmTile(Math::ibounds2 const& sampleBounds) noexcept -> Scope<FilmTile>;
		auto mergeFilmTile(Scope<FilmTile> tile) noexcept -> void;

		struct Pixel {
			Math::vec3 xyz;
			// the sum of filter weight values for the sample contributions to the pixel
			float filterWeightSum = 0;
			Parallelism::AtomicFloat splatXYZ[3];
			float pad;
		};

		auto getPixel(Math::ipoint2 const& p) noexcept -> Pixel&;

		auto setImage(Spectrum const* img) const noexcept -> void;

		auto addSplat(Math::point2 const& p, Spectrum const& v) noexcept -> void;

		auto writeImage(Image::Image<Image::COLOR_R8G8B8_UINT>& image, float splatScale = 1.f) noexcept -> void;

		/** Length of the diagnoal of the film's physical area in meters*/
		float const diagonal;
		

		float const scale;

		/** With typical filter every sample may contribute to 16 or more pixel in the final image */
		std::unique_ptr<Filter> filter;
		std::string const filename;

	private:
		std::unique_ptr<Pixel[]> pixels;

		/*
		* Assumpt that filter is defined such that f(x,y) = f(|x|,|y|).
		* Table only hold values for only the positive quadrant of filter offsets.
		*/
		static constexpr int filterTableWidth = 16;
		float filterTable[filterTableWidth * filterTableWidth];

		std::mutex mutex;
	};

	export struct FilmTilePixel {
		Spectrum contribSum;
		float filterWeightSum = 0.f;
	};

	export struct FilmTile {
		FilmTile(Math::ibounds2 const& pixelBounds, Math::vec2 const& filterRadius, float const* filterTable, int filterTableSize);

		auto addSample(Math::point2 const& pFilm, Spectrum const& L, float sampleWeight) noexcept -> void;
		auto getPixel(Math::ipoint2 const& p) noexcept -> FilmTilePixel&;
		auto getPixelBounds() const noexcept -> Math::ibounds2;

		Math::ibounds2 const pixelBounds;
		Math::vec2 const filterRadius, invFilterRadius;
		float const* filterTable;
		int const filterTableSize;
		std::vector<FilmTilePixel> pixels;
	};
}
