module;
#include <string>
#include <memory>
export module Tracer.Film:Film;
import Math.Geometry;
import Tracer.Filter;

namespace SIByL::Tracer
{
	export struct Film
	{
		/**
		* @param cropWindow: Specify a subset of image to render, In NDC spacem with each coordinate ranging from 0-1
		* @param diagonal  : Length of the diagnoal of the film's physical area in millimeters
		* @param filename  : Filename for the output image
		*/
		Film(Math::ipoint2 const& resolution, Math::bounds2 const& cropWindow,
			std::unique_ptr<Filter> filt, float diagonal,
			std::string const& filename, float scale);

		/** Overall resolution of the image in pixels */
		Math::ipoint2 const fullResolution;
		
		/** Length of the diagnoal of the film's physical area in meters*/
		float const diagonal;
		
		Math::ibounds2 croppedPixelBounds;

		float const scale;

		std::unique_ptr<Filter> filter;
		std::string const filename;
	};
}
