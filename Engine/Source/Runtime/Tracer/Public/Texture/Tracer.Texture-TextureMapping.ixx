export module Tracer.Texture:TextureMapping;
import Math.Vector;
import Math.Geometry;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	/** Provides an interface for computing 2D coordinates. */
	export struct TextureMapping2D {
		/**
		* Given the surface interaction at the shading point and returns the (s,t) texture coordinates.
		* Also returns estimated change in texture coordinates with respect to pixel coordinates, which
		* could help determining the (s,t) sampling rate and filter accordingly.
		* @param dstdx is the estimated change in (s,t) with respect to pixel x coordniates.
		* @param dstdy is the estimated change in (s,t) with respect to pixel y coordniates.
		*/
		virtual auto map(SurfaceInteraction const& si, 
			Math::vec2* dstdx, Math::vec2* dstdy) const noexcept -> Math::point2 = 0;
	};

	/** Provides an interface for computing 3D coordinates. */
	export struct TextureMapping3D {

	};

	/**
	* Uses the (u,v) coordinates in the SurfaceInteraction to compute the texture coordinates.
	* Their values can be offset and scaled with user-supplied values in each dimension.
	*/
	export struct UVMapping2D :public TextureMapping2D {
		/**
		* Initialize with scale-and-shift parameters
		* @param su/sv: scale on u/v
		* @param du/dv: offset on u/v
		*/
		UVMapping2D(float su, float sv, float du, float dv)
			:su(su), sv(sv), du(du), dv(dv) {}

		virtual auto map(SurfaceInteraction const& si,
			Math::vec2* dstdx, Math::vec2* dstdy) const noexcept -> Math::point2 override 
		{
			// compute texture diffrentials for 2D (u,v) mapping
			*dstdx = Math::vec2(su * si.dudx, sv * si.dvdx);
			*dstdy = Math::vec2(su * si.dudy, sv * si.dvdy);
			return Math::point2(su * si.uv.u + du, sv * si.uv.v + dv);
		}

	private:
		float const su, sv, du, dv;
	};

}