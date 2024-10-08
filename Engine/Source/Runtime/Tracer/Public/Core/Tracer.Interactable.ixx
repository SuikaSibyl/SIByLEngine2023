module;
#include <cstdint>
#include <vector>
export module Tracer.Interactable;
import SE.Core.Memory;
import SE.Math.Geometric;
import Tracer.Ray;
import Tracer.BxDF;
import Tracer.Base;
import Tracer.Medium;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	// **************************************************************************************
	// | Interactable:																		|
	// |************************************************************************************|
	// | This module includes many importance classes that are interactable to ray.			|
	// | Due to heavy recursive-dependency and poor forward declaration in C++ 20 modules,	|
	// | all these classes are declared together here.										|
	// **************************************************************************************
	// Interaction categories
	struct Interaction;
	struct SurfaceInteraction;
	struct MediumInteraction;
	// Material categories
	struct BSDF;
	struct BSSRDF;
	struct Material;
	template <class T> struct Texture;
	// Geometries categories
	struct Shape;
	struct Primitive;
	// Light & AreaLight
	struct Light;
	struct AreaLight;
	// the Scene
	struct Scene;

	// ****************************
	// | Interaction Categories	  |
	// ****************************
	/**
	* 
	*/
	export struct Interaction
	{
		Interaction() = default;
		Interaction(Math::point3 const& p, Math::normal3 const& n, Math::vec3 const& pError,
			Math::vec3 const& wo, float time,
			MediumInterface const& mediumInterface)
			: p(p), time(time), pError(pError), wo(wo), n(n),
			mediumInterface(mediumInterface) { }

		Interaction(Math::point3 const& p, Math::vec3 const& wo, float time, MediumInterface const& mediumInterface)
			: p(p), time(time), wo(wo), mediumInterface(mediumInterface) { }

		Interaction(Math::point3 const& p, float time, MediumInterface const& mediumInterface)
			: p(p), time(time), mediumInterface(mediumInterface) { }

		auto isSurfaceInteraction() const noexcept -> bool;
		auto isMediumInteraction() const noexcept -> bool;

		auto spawnRay(Math::vec3 const& d) const noexcept -> Ray;
		//TODO
		auto spawnRayTo(Math::point3 const& p) const noexcept -> Ray { return Ray{}; }
		//TODO
		auto spawnRayTo(Interaction const& i) const noexcept -> Ray { return Ray{}; }

		auto getMedium(Math::vec3 const& w) const noexcept -> Medium const*;

		/** interaction position */
		Math::point3 p;
		/** interaction time */
		float time;
		/** a conservative bound on floating-point error of point p, when computed by ray intersection */
		Math::vec3 pError;
		/** if interaction lies along a ray, wo is the negative ray direction */
		Math::vec3 wo;
		/** surface normal of point p */
		Math::normal3 n;
		/** the scattering media at the point (if any) */
		MediumInterface mediumInterface;
	};

	/**
	* Represent local information at a point on a 2D surface.
	* Supply enough information about the surface point.
	*/
	export struct SurfaceInteraction :public Interaction
	{
		SurfaceInteraction();
		SurfaceInteraction(Math::point3 const& p, Math::point3 const& pError,
			Math::point2 const& uv, Math::vec3 const& wo,
			Math::vec3 const& dpdu, Math::vec3 const& dpdv,
			Math::normal3 const& dndu, Math::vec3 const& dndv,
			float time, Shape const* shape);

		/**
		* Set shading geometry, a second instance of geometry information,
		* generated by bump mapping or interpolated per-vertex normals with triangles.
		* @see SurfaceInteraction::shading
		*/
		auto setShadingGeometry(Math::vec3 const& dpdus, Math::vec3 const& dpdvs,
			Math::normal3 const& dndus, Math::normal3 const& dndvs, bool orientationIsAuthoritative) noexcept -> void;

		auto computeScatteringFunctions(RayDifferential const& ray, Core::MemoryArena& arena, bool allowMultipleLobes = false, TransportMode mode = TransportMode::Radiance) noexcept -> void;

		/**
		* Compute partial derivatives for mipmap-based texture sampling.
		* It is called by computeScatteringFunctions().
		*/
		auto computeDifferentials(RayDifferential const& ray) const noexcept -> void;

		auto Le(Math::vec3 const& w) const noexcept -> Spectrum;

		/** parameterizeation of the surface */
		Math::point2 uv;
		/** the parametric partial derivatives of the point ∂p/∂u and ∂p/∂v */
		Math::vec3 dpdu, dpdv;
		/** the differential change in surface normal as we move u and v along the surface */
		Math::normal3 dndu, dndv;

		/** the Shape that the point lies on */
		Shape const* shape = nullptr;
		/** the Primitive that the ray hits */
		Primitive const* primitive = nullptr;

		/* the partial derivatives of world space positions to pixel position ∂p/∂x & ∂p/∂y*/
		mutable Math::vec3 dpdx, dpdy;
		/* the partial derivatives of texture coordinates to pixel position ∂p/∂x & ∂p/∂y*/
		mutable float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;

		BSDF* bsdf = nullptr;
		BSSRDF* bssrdf = nullptr;

		/**
		* The shading geometry, a second instance of geometry information.
		* Which can be generated by bump mapping or interpolated per-vertex normals with triangles.
		*/
		struct {
			Math::normal3 n;
			Math::vec3 dpdu, dpdv;
			Math::normal3 dndu, dndv;
		} shading;
	};

	/**
	* Represent points where light scatters in participating media.
	*/
	export struct MediumInteraction :public Interaction
	{
		MediumInteraction(Math::point3 const& p, Math::vec3 const& wo, float time,
			Medium const* medium, PhaseFunction const* phase)
			:Interaction(p, wo, time, medium), phase(phase) {}

		PhaseFunction const* phase;
	};

	// ****************************
	// | Material Categories	  |
	// ****************************
	/**
	* BSDF struct represents a collection of BRDFs and BTDFs.
	* BSDF implementation also deal with shading normal problem.
	*/
	export struct BSDF
	{
		/**
		* Compute an orthonormal coordinate system with the shading normal as one of the axes.
		*
		* @param si: information about the differential geometry at the point on a surface
		* @param eta: relative index of refraction over the boundary
		*/
		BSDF(SurfaceInteraction const& si, float eta = 1);

		auto add(BxDF* b) noexcept -> void;

		/** Return the number of BxDFs stored by the BSDF that match a particular set of BxDFType flags */
		auto numComponents(BxDF::Type flags = BxDF::BSDF_ALL) const noexcept -> int;

		auto worldToLocal(Math::vec3 const& v) const noexcept -> Math::vec3;

		auto localToWorld(Math::vec3 const& v) const noexcept -> Math::vec3;

		auto f(Math::vec3 const& woW, Math::vec3 const& wiW, BxDF::Type flags = BxDF::Type::BSDF_ALL) const noexcept -> Spectrum;

		// -------------------------------
		// Aggregate Behavior of the 4D BSDF
		// -------------------------------
		/**
		* Computes hemispherical-directional reflectance ρ_hd.
		* Gives the total reflection in a given direction due to constant inllumination over the hemisphere.
		* Or equivalently total reflection over the hemisphere due to light from a given direction.
		* @param nSamples	: the number of Monte Carlo samples to take to approximate ρ_hd
		* @param samples	: the samples for using Monte Carlo to approximate ρ_hd
		*/
		auto rho(Math::vec3 const& wo, int nSamples, Math::point2 const* samples, BxDF::Type flags = BxDF::BSDF_ALL) const noexcept -> Spectrum;

		/**
		* Computes hemispherical-hemispherical reflectance ρ_hh.
		* Is the fraction of incident light reflected by a surface when the incident light is the same from all directions.
		* @param nSamples	: the number of Monte Carlo samples to take to approximate ρ_hh
		* @param u1/u2		: the samples for using Monte Carlo to approximate ρ_hh
		*/
		auto rho(int nSamples, Math::point2 const* samples1, Math::point2 const* samples2, BxDF::Type flags = BxDF::Type::BSDF_ALL) const noexcept -> Spectrum;

		// -------------------------------
		// Sampling Reflection Functions
		// -------------------------------
		/**
		* Called by Integrators to sample according to BSDF's distribution.
		* Calls the individual BxDF::sample_f() methods to generate samples.
		* @param woWorld	: outgoing direction passed, in world space
		* @param wiWorld	: incident direction returned, in world space
		*/
		auto sample_f(Math::vec3 const& woWorld, Math::vec3* wiWorld, Math::point2 const& u, float* pdf, BxDF::Type type, BxDF::Type* sampledType = nullptr) const noexcept -> Spectrum;

		/**
		* Return the PDF for a given pair of directions
		* @param woWorld	: outgoing direction passed, in world space
		* @param wiWorld	: incident direction passed, in world space
		*/
		auto pdf(Math::vec3 const& wo, Math::vec3 const& wi, BxDF::Type flags = BxDF::BSDF_ALL) const noexcept -> float;

		float const eta;
		Math::normal3 ns, ng;
		Math::vec3 ss, ts;

		static constexpr int maxBxDFs = 8;
		int nBxDFs = 0;
		BxDF* bxdfs[maxBxDFs];

	private:
		~BSDF() {}
	};

	/**
	* BSSRDF
	*/
	export struct BSSRDF
	{
		/**
		* @param po is the current outgoing surface interaction
		* @param eta is the index of refraction of the scattering medium, which is assuemd to be a constant
		*/
		BSSRDF(SurfaceInteraction const& po, float eta) :po(po), eta(eta) {}

		/**
		* Evaluate the eight-dimensional distribution function,
		* which qualifies the ratio of differential radiance at point p_o in direction ω_o
		* to the incident differential flux at p_i from direction ω_i.
		*/
		virtual auto s(SurfaceInteraction const& pi, Math::vec3 const& wi) noexcept -> Spectrum { return Spectrum{}; }

		SurfaceInteraction const& po;
		float eta;
	};

	export inline auto fresnelMoment1(float invEta) noexcept -> float;
	export inline auto fresnelMoment2(float invEta) noexcept -> float;

	/*
	* Primitve is the bridge between the geometry processing & shading subsystem of pbrt
	*/
	export struct Primitive
	{
		/** A box that enclose the primitive's geometry in world space */
		virtual auto worldBound() const noexcept -> Math::bounds3 = 0;

		/**
		* Update Ray::tMax with distance value if an intersection is found.
		* Initialize additional SurfaceInteraction member variables.
		*/
		virtual auto intersect(Ray const& r, SurfaceInteraction* i) const noexcept -> bool = 0;
		virtual auto intersectP(Ray const& r) const noexcept -> bool = 0;

		/** Describes the primitive's emission distribution if it's a light source */
		virtual auto getAreaLight() const noexcept -> AreaLight const* = 0;
		/** Return a pointer to the material instance assigned to the primitive */
		virtual auto getMaterial() const noexcept -> Material const* = 0;

		/**
		* Initialize representations of the light-scattering properties of
		* the material at the intersection point on the surface.
		*/
		virtual auto computeScatteringFunctions(
			SurfaceInteraction* isec,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void = 0;
	};


	export struct VisibilityTester
	{
		VisibilityTester() {}
		VisibilityTester(Interaction const& p0, Interaction const& p1) :_p0(p0), _p1(p1) {}

		auto p0() const noexcept -> Interaction const& { return _p0; }
		auto p1() const noexcept -> Interaction const& { return _p1; }

		auto unoccluded(Scene const& scene) const noexcept -> bool;

		auto Tr(Scene const& scene, Sampler& sampler) const noexcept -> Spectrum;

		Interaction _p0, _p1;
	};

	export enum struct LightFlags :uint32_t {
		DeltaPosition = 1,
		DeltaDirection = 2,
		Area = 4,
		Infinite = 8,
	};

	export inline auto isDeltaLight(int flags) noexcept -> bool {
		return flags * (int)LightFlags::DeltaPosition || flags & (int)LightFlags::DeltaDirection;
	}

	export struct Light
	{
		Light(int flags, Math::Transform const& LightToWorld,
			MediumInterface const& mediumInterface, int nSamples = 1);

		/** Indicates the fundamental light source type. */
		int const flags;

		int const nSamples;

		MediumInterface const mediumInterface;

		virtual auto Le(RayDifferential const& ray) const noexcept -> Spectrum { return Spectrum{ 0.f }; }

		virtual auto preprocess(Scene const& scene) noexcept -> void {}

		virtual auto power() const noexcept -> Spectrum = 0;

		// -------------------------------
		// Sampling Light Sources
		// -------------------------------
		/**
		* Samples an incident direction at a point in the scene along which
		* illumination from the light may be arriving.
		* @param ref	: provides the world space position of a reference point in the scene
		*				  and time with it (to resolve visibility)
		* @param u		: provides a 2D sample value for sampling the light source
		* @param wi		: return the incident direction sampled
		* @param pdf	: return the PDF for sampling the chosen direction
		* @return the radiance arriving at the point at the time due to the light
		*/
		virtual auto sample_Li(Interaction const& ref, Math::point2 const& u,
			Math::vec3* wi, float* pdf, VisibilityTester* vis) const noexcept -> Spectrum = 0;

		/**
		* Return the probability density with respect to solid angle for sample_Li() samples
		* the direction wi from the reference point ref.
		*/
		virtual auto pdf_Li(Interaction const& ref, Math::vec3 const& wi) const noexcept -> float = 0;

		/**
		* Returns a light-carrying ray leaving the light source.
		* TODO
		*/
		virtual auto sample_Le() const noexcept -> Spectrum = 0;

		// -------------------------------
		// Query Properties
		// -------------------------------
		/** tell whether the light has delta distribution */
		virtual inline auto isDeltaLight(int flags) noexcept -> bool {
			return flags & (int)LightFlags::DeltaPosition || flags & (int)LightFlags::DeltaDirection;
		}

		/**
		* Light's coordinate system with respect to world space.
		* Could implement a light assuming a particular coordinate system,
		* and use transform to place it at arbitrary position & orientations.
		*/
		Math::Transform lightToWorld, worldToLight;
	};

	export struct AreaLight :public Light
	{
		virtual auto L(Interaction const& intr, Math::vec3 const& w) const noexcept -> Spectrum = 0;
	};

	export struct Material
	{
		virtual auto computeScatteringFunctions(
			SurfaceInteraction* isec,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void = 0;

		auto bump(Texture<float>* d, SurfaceInteraction* si) noexcept -> void;
	};

	export struct Scene
	{
		Scene(Primitive* aggregate, std::vector<Light*> const& lights);

		auto getWorldBound() const noexcept -> Math::bounds3 const&;

		/**
		* Traces the given ray into the scene and returns a bool value
		* indicating whether the ray intersected any of the primitives.
		* If so, fills the provided SurfaceInteraction structure with
		* information about the closet intersection point along the ray.
		*/
		auto intersect(Ray const& ray, SurfaceInteraction* isect) const noexcept -> bool;

		/*
		* Checks for the existence of intersections along the ray but does
		* not return any information about those intersections. It is generally
		* more efficient and regularly used for shadow rays.
		*/
		auto intersectP(Ray const& ray) const noexcept -> bool;

		auto intersectTr(Ray ray, Sampler& sampler, SurfaceInteraction* isect, Spectrum* Tr) const noexcept -> bool;

		std::vector<Light*> lights;
		Primitive* aggregate;
		Math::bounds3 worldBound;
	};

	/**
	* A general Shape interface.
	*/
	export struct Shape
	{
		Shape(Math::Transform const* objectToWorld, Math::Transform const* worldToObject, bool reverseOrientation)
			: objectToWorld(objectToWorld)
			, worldToObject(worldToObject)
			, reverseOrientation(reverseOrientation)
			, transformSwapsHandedness(objectToWorld->swapsHandness())
		{}

		/** return a bounding box in shape's object space */
		virtual auto objectBound() const noexcept -> Math::bounds3 = 0;
		/** return a bounding box in world space */
		auto worldBound() const noexcept -> Math::bounds3;

		/**
		* Returns geometric information about a single ray–shape intersection,
		* corresponding to the first intersection, if any, in the (0, tMax)
		* parametric range along the ray;
		* @param ray: input ray is in world space;
		* @param tHit: the parametric distance along the ray;
		* @param isect: a capture of local geometric properties of a surface, in world space;
		* @param testAlphaTexture: indicate whether perfirm a texture base surface cutting;
		*/
		virtual auto intersect(
			Ray const& ray,
			float* tHit,
			SurfaceInteraction* isect,
			bool testAlphaTexture = true) const -> bool = 0;

		/**
		* A predicate version of intersect(), which only determines whether or not
		* an intersection occurs, without returning any details about the intersection.
		* The default impl directly call intersect, which is wastefull.
		* @see intersect()
		*/
		virtual auto intersectP(
			Ray const& ray,
			bool testAlphaTexture = true) const  -> bool;

		/**
		* Compute the surface area of a shape in object space.
		* It is necessary when use Shapes as area lights.
		*/
		virtual auto area() const noexcept -> float = 0;

		// -------------------------------
		// Sampling Shape
		// -------------------------------
		/**
		* Chooses points on the surface of the shape using a sampling distribution
		* with respect to surface area. Returns the local geometric information about
		* the sampled point in an Interaction.
		*/
		virtual auto sample(Math::point2 const& u) const noexcept -> Interaction = 0;
		/**
		* Return the pdf when the interaction is sampled by sample() function.
		* Default implementation assume sample() uniformly sample points on the surface.
		*/
		virtual auto pdf(Interaction const&) const noexcept -> float;
		/**
		* Takes the point from which the surface of the shape is
		* being integrated over as a parameter.
		* Which is particularly useful for lighting.
		*/
		virtual auto sample(const Interaction& ref, Math::point2 const& u) const noexcept -> Interaction;
		/**
		* The second sample function uses a density with respect to solid angle from the reference point ref.
		* The default implementation transforms the density from one defined over area to one defined ovver solid angle.
		*/
		auto pdf(const Interaction& ref, Math::vec3 const& wi) const noexcept -> float;

		/**
		* All shapes are defined in object coordinate,
		* use object-to-world Transform to present Transformation
		*/
		Math::Transform const* objectToWorld;
		Math::Transform const* worldToObject;
		/** Whether surface normal directions should be reversed from default */
		bool const reverseOrientation;
		/** The value of Transform::SwapsHandedness() of object-to-world Transform */
		bool const transformSwapsHandedness;
	};

	/** Texture is a function that maps points in some domain to values in some other domian. */
	export template <class T>
	struct Texture {
		/** Evaluate texture's value at the point being shaded of the SurfaceInteraction.*/
		virtual auto evaluate(SurfaceInteraction const&) const noexcept -> T = 0;
	};
}

namespace SIByL::Math
{
	export inline auto operator*(Transform const& t, Tracer::SurfaceInteraction const& si)->Tracer::SurfaceInteraction;
	export inline auto operator*(AnimatedTransform const& t, Tracer::SurfaceInteraction const& si)->Tracer::SurfaceInteraction;
}
