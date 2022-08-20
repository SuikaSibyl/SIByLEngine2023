module;
#include <cmath>
#include <algorithm>
module Tracer.Integrator:SamplerIntegrator;
import Tracer.Integrator;
import Core.Memory;
import Math.Vector;
import Math.Geometry;
import Parallelism.Parallel;
import Tracer.Ray;
import Tracer.Camera;
import Tracer.Sampler;
import Tracer.Spectrum;
import Tracer.Film;
import Tracer.BxDF;

namespace SIByL::Tracer
{
	auto SamplerIntegrator::render(Scene const& scene) noexcept -> void
	{
		preprocess(scene);
		// render image tiles in parallel
		// compute number of tiles, nTiles, to use for parallel rendering
		Math::ibounds2 sampleBounds = camera->film->getSampleBounds();
		Math::ivec2 sampleExtent = sampleBounds.diagonal();
		int const tileSize = 16;
		Math::ipoint2 nTiles((sampleExtent.x + tileSize - 1) / tileSize,
							 (sampleExtent.y + tileSize - 1) / tileSize);

		Parallelism::ParallelFor2D([&](Math::ipoint2 tile) {
			// Render section of image corresponding to tile
			// allocate MemoryArena for tile
			Core::MemoryArena arena;
			// get sampler instance for tile
			int seed = tile.y * nTiles.x + tile.x;
			Scope<Sampler> tileSampler = sampler->clone(seed);
			// compute sample bounds for tile
			int x0 = sampleBounds.pMin.x + tile.x * tileSize;
			int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
			int y0 = sampleBounds.pMin.y + tile.y * tileSize;
			int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
			Math::ibounds2 tileBounds(Math::ipoint2{ x0,y0 }, Math::ipoint2{ x1,y1 });
			// get FilmTile for tile
			Scope<FilmTile> filmTile = camera->film->getFilmTile(tileBounds);
			// loop over pixels in tile to render them
			for (Math::ipoint2 pixel : tileBounds) {
				tileSampler->startPixel(pixel);
				do {
					// initialzie CameraSample for current sample
					CameraSample cameraSample = tileSampler->getCameraSample(pixel);
					// generate camera ray for current sample
					RayDifferential ray;
					float rayWeight = camera->generateRayDifferential(cameraSample, &ray);
					ray.scaleDifferentials(1.f / (float)std::sqrt(tileSampler->samplesPerPixel));
					// evaluate radiance along camera ray
					Spectrum L(0.f);
					if (rayWeight > 0)
						L = Li(ray, scene, *tileSampler, arena, 0);
					// add camera ray's contribution to image
					filmTile->addSample(cameraSample.pFilm, L, rayWeight);
					// free MemoryArena memory from computing image sample value
					arena.reset();
				} while (tileSampler->startNextSample());
			}
			// merge image tiles into Film
			camera->film->mergeFilmTile(std::move(filmTile));
			}, nTiles);
		// save final image after rendering
		camera->film->writeImage();
	}

	auto SamplerIntegrator::specularReflect(RayDifferential const& ray, SurfaceInteraction const& isect, Scene const& scene, Sampler& sampler, Core::MemoryArena& arena, int depth) const noexcept -> Spectrum {
		// compute specular reflection direction wi and BSDF value
		Math::vec3 wo = isect.wo, wi;
		float pdf;
		BxDF::Type type = BxDF::Type(BxDF::BSDF_REFLECTION | BxDF::BSDF_SPECULAR);
		Spectrum f = isect.bsdf->sample_f(wo, &wi, sampler.get2D(), &pdf, type);
		// return contribution of specular reflection
		Math::normal3 const& ns = isect.shading.n;
		if (pdf > 0 && !f.isBlack() && Math::absDot(wi, ns) != 0) {
			// compute ray differential rd for specular reflection
			RayDifferential rd = isect.spawnRay(wi);
			if (ray.hasDifferentials) {
				rd.hasDifferentials = true;
				rd.rxOrigin = isect.p + isect.dpdx;
				rd.ryOrigin = isect.p + isect.dpdy;
				// Compute differential reflected directions
				Math::normal3 dndx = isect.shading.dndu * isect.dudx + isect.shading.dndv * isect.dvdx;
				Math::normal3 dndy = isect.shading.dndu * isect.dudy + isect.shading.dndv * isect.dvdy;
				Math::vec3 dwodx = -ray.rxDirection - wo, dwody = -ray.ryDirection - wo;
				float dDNdx = Math::dot(dwodx, ns) + Math::dot(wo, dndx);
				float dDNdy = Math::dot(dwody, ns) + Math::dot(wo, dndy);
				rd.rxDirection = wi - dwodx + 2.f * Math::vec3(Math::dot(wo, ns) * dndx + dDNdx * ns);
				rd.ryDirection = wi - dwody + 2.f * Math::vec3(Math::dot(wo, ns) * dndy + dDNdy * ns);
			}
			return f * Li(rd, scene, sampler, arena, depth + 1) * Math::absDot(wi, ns) / pdf;
		}
		else
			return Spectrum{ 0.f };
	}

	auto SamplerIntegrator::specularTransmit(RayDifferential const& ray, SurfaceInteraction const& isect, Scene const& scene, Sampler& sampler, Core::MemoryArena& arena, int depth) const noexcept -> Spectrum {
		// compute transmission direction wi and BSDF value
		Math::vec3 wo = isect.wo, wi;
		float pdf;
		BxDF::Type type = BxDF::Type(BxDF::BSDF_TRANSMISSION);
		Spectrum f = isect.bsdf->sample_f(wo, &wi, sampler.get2D(), &pdf, type);
		// return contribution of specular reflection
		Math::normal3 const& ns = isect.shading.n;
		if (pdf > 0 && !f.isBlack() && Math::absDot(wi, ns) != 0) {
			// compute ray differential rd for specular reflection
			RayDifferential rd = isect.spawnRay(wi);
			if (ray.hasDifferentials) {
				rd.hasDifferentials = true;
				rd.rxOrigin = isect.p + isect.dpdx;
				rd.ryOrigin = isect.p + isect.dpdy;
				// Compute differential reflected directions
				Math::normal3 dndx = isect.shading.dndu * isect.dudx + isect.shading.dndv * isect.dvdx;
				Math::normal3 dndy = isect.shading.dndu * isect.dudy + isect.shading.dndv * isect.dvdy;
				Math::vec3 dwodx = -ray.rxDirection - wo, dwody = -ray.ryDirection - wo;
				float dDNdx = Math::dot(dwodx, ns) + Math::dot(wo, dndx);
				float dDNdy = Math::dot(dwody, ns) + Math::dot(wo, dndy);
				rd.rxDirection = wi - dwodx + 2.f * Math::vec3(Math::dot(wo, ns) * dndx + dDNdx * ns);
				rd.ryDirection = wi - dwody + 2.f * Math::vec3(Math::dot(wo, ns) * dndy + dDNdy * ns);
			}
			return f * Li(rd, scene, sampler, arena, depth + 1) * Math::absDot(wi, ns) / pdf;
		}
		else
			return Spectrum{ 0.f };
	}

}