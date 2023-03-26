#ifndef _SRENDERER_MMLT_GLSL_COMMON_HEADER_
#define _SRENDERER_MMLT_GLSL_COMMON_HEADER_

#include "../include/common_trace.h"
#include "../include/common_rgen.h"

#include "mmlt_config.h"
#include "../bdpt/bdpt_common.h"

//////////////////////////////////////////////////////////////////////
//              Random Variable Mutation Strategies                 //
//////////////////////////////////////////////////////////////////////

/** Uniform generate a vec4 in [0,1]. **/
vec4 UniformVec4(
    inout uint RNG
) {
    vec4 vo;
    vo.x = UniformFloat(RNG);
    vo.y = UniformFloat(RNG);
    vo.z = UniformFloat(RNG);
    vo.w = UniformFloat(RNG);
    return vo;
}

/**
* >> SRenderer :
* Uniformly fill a sample stream.
*/
void UniformSampleStream(
    out vec4 sample_streams[num_states_vec4],
    in uint RNG
) {
    for(int i=0; i<num_states_vec4; ++i)
        sample_streams[i] = UniformVec4(RNG);
}

/**
* >> SRenderer :
* Small step float mutation.
* It generally generate a random step smaller than s2.
* More likely to choose smaller values,
* the decay of inverse transform is exponentially decreased related to s1.
*
* @param value: value to mutate from
* @param s1:    should be smaller than s2, larger s1 means slower decay.
* @param s2:    the largest step length.
* @ref:         see smallpssmlt by Toshiya Hachisuka
*/
float perturbFloat(
    in const float value,
    in const float s1,
    in const float s2,
    inout uint     RNG
) {
	float result;
	float r = UniformFloat(RNG);
	if (r < 0.5) {  // mutate to positive direction
		r = r * 2.0;
		result = value + s2 * exp(-log(s2 / s1) * r);
        if (result > 1.0) result -= 1.0;
	} else {        // mutate to negative direction
		r = (r - 0.5) * 2.0;
		result = value - s2 * exp(-log(s2 / s1) * r);
        if (result < 0.0) result += 1.0;
	}
	return result;
}

/**
* >> SRenderer :
* Vec4 version of previous function.
*/
vec4 perturbVec4(
    in const vec4  value,
    in const float s1,
    in const float s2,
    inout uint     RNG
) {
    vec4 o;
    o.x = perturbFloat(value.x, s1, s2, RNG);
    o.y = perturbFloat(value.y, s1, s2, RNG);
    o.z = perturbFloat(value.z, s1, s2, RNG);
    o.w = perturbFloat(value.w, s1, s2, RNG);
    return o;
}


/**
* >> SRenderer :
* @param input_streams:  stream to mutate from
* @param sample_streams: stream that mutate to
* @ret:                  whether use large step mutation. 1=yes, 0=no.
*/
int mutation_v0(
    in const vec4  input_streams[num_states_vec4],
    out      vec4  sample_streams[num_states_vec4],
    in const ivec2 canvas_size,
    inout    uint  RNG
) {
    const bool  largeStep = UniformFloat(RNG) < large_step_probability;
    
    const float rPixel = 2.0 / float(canvas_size.x + canvas_size.y);
    const float rUnifm = 1.0 / 1024.0;

    if(largeStep){  // use large step mutation
        for(int i=0; i<num_states_vec4; ++i)
            sample_streams[i] = UniformVec4(RNG);
        return 1;
    }
    else { // use small step mutation
        for(int i=0; i<num_states_vec4; ++i) {
            // if (i==offset_camera_subpath) { // special handling camera
            //     sample_streams[i].x = perturbFloat(input_streams[i].x, rPixel, 0.1, RNG);
            //     sample_streams[i].y = perturbFloat(input_streams[i].y, rPixel, 0.1, RNG);
            //     sample_streams[i].z = perturbFloat(input_streams[i].z, rUnifm, 1.0 / 64.0, RNG);
            //     sample_streams[i].w = perturbFloat(input_streams[i].w, rUnifm, 1.0 / 64.0, RNG);
            // }
            // else
                sample_streams[i] = perturbVec4(input_streams[i], rUnifm, 1.0 / 64.0, RNG);
        }
        return 0;
    }
}

//////////////////////////////////////////////////////////////////////
//              Deterministic version of BDPT APIs                  //
//////////////////////////////////////////////////////////////////////

/**
* pathRandomWalk() traces paths starting at an initial vertex.
* Givens a start ray, it will takes care of sampling the following vertices and initializing the path array.
* Deterministic version for PSSMLT
* @ret: the number of bounces in random walk.
*/
int pathRandomWalk(
    inout BDPTPath  path,
    in Ray ray,
    in const int maxDepth,
    in  vec4  sample_stream[num_states_vec4],
    in  int   stream_offset,
    in const float pdf,
    in const bool isImportance
) {
    if (maxDepth == 0) return 0;
    int bounces = 0;
    // At the beginning of each iteration
    // pdfFwd records the probability per unit solid angle of the sampled ray direction ray.d
    float pdfFwd = pdf;
    // On the other hand, pdfRev denotes the reverse probability at the end of each iteration,
    // that is, the density of the opposite light transport mode per unit solid angle along the same ray segment.
    float pdfRev = 0;

    // Declare variables for forward and reverse probability densities
    vec3 beta = path.x[0].beta;

    while(true) {
        // Trace the ray
        // Attempt to create the next subpath vertex in path
        
        // Handle surface interaction for path generation
        if (!intersect(ray, path.x[path.n].isect)) {
            // TODO :: Capture escaped rays when tracing from the camera
            return bounces;
        }
        // Initialize vertex with surface intersection information
        // the solid angle density pdfFwd is converted to an area density before being stored in Vertex::pdfFwd
        path.x[path.n].pdfFwd = convertDensity(pdfFwd, path.x[path.n-1], path.x[path.n]);
        path.x[path.n].pdfRev = 0;
        path.x[path.n].type = enum_vertex_type_surface;
        const vec3 dir_view = -ray.direction;
        if(isImportance)
            beta *= correctShadingNormal_left(path.x[path.n].isect, dir_view);
        path.x[path.n].beta = beta;

        // fetch rvecs
        vec4 rvec = sample_stream[stream_offset + bounces];
        // Sample BSDF at current vertex and compute reverse probability
        vec2 bsdf_uv = rvec.xy;
        vec3 dir_bsdf = bsdf_sample(
            path.x[path.n].isect,
            dir_view,
            bsdf_uv,
            rvec.z);

        vec3 bsdf = bsdf_eval(path.x[path.n].isect, dir_bsdf);
        pdfFwd = bsdf_pdf(path.x[path.n].isect, dir_bsdf);
        pdfRev = bsdf_pdf(path.x[path.n].isect, dir_bsdf, path.x[path.n].isect.wo);
        // Compute reverse area density at preceding vertex
        path.x[path.n-1].pdfRev = convertDensity(pdfRev, path.x[path.n], path.x[path.n-1]);

        if (pdfFwd <= 0) {
            // Numerical issue -- we generated some invalid rays.
            break;
        }
        beta *= bsdf / pdfFwd;
        if(isImportance)
            beta *= correctShadingNormal_right(path.x[path.n].isect, dir_bsdf);
        
        ray = spawnRay(path.x[path.n].isect, dir_bsdf);
        // Update loop
        ++path.n;
        ++bounces;

        if (bounces >= maxDepth) {
            break;
        }
    }
    return bounces;
}

/**
* Generate camera subpath.
* Deterministic version for PSSMLT
* @ret: return the number of vertices in the subpath.
*/
int generateCameraSubpath(
    in  int   maxCameraEvents,
    in  vec2  pixel,
    in  ivec2 frameDim,
    out BDPTPath path,
    in  vec4  sample_stream[num_states_vec4]
) {
    initialize(path);
    // No path to build
    if (maxCameraEvents == 0)
        return 0;
    // Sample initial ray for camera subpath
    Ray ray = generateRay(pixel, frameDim, false);
    // Generate first vertex on camera subpath
    path.x[0].beta = vec3(1);
    path.x[0].isect.position = ray.origin;
    path.x[0].isect.geometric_normal = normalize(globalUniform.cameraData.cameraW);
    path.x[0].pdfFwd = 0;
    path.x[0].pdfRev = 0;
    path.x[0].type = enum_vertex_type_camera;
    path.n++;
    //
    const float cosTheta = abs_dot(path.x[0].isect.geometric_normal, ray.direction);
    const float pdf_dir = 1 / (globalUniform.cameraData.rectArea * cosTheta * cosTheta * cosTheta);;
    // Start random walk
    pathRandomWalk(path, ray, maxCameraEvents-1, sample_stream, offset_camera_subpath+1, pdf_dir, false);
    return path.n;
}

/**
* Generate light subpath.
* Deterministic version for PSSMLT
* @ret: return the number of vertices in the subpath.
*/
int generateLightSubpath(
    in  int maxDepth,
    out BDPTPath path,
    in  vec4  sample_stream[num_states_vec4]
) {
    initialize(path);
    // No path to build
    if (maxDepth == 0) return 0;
    // Fetch samples for light sampling
    // It need one more rvec, in total 2.
    vec4 rvec_0 = sample_stream[offset_light_subpath + 0];
    vec4 rvec_x = sample_stream[offset_light_subpath + max_events];
    // Sample initial ray for light subpath
    LightSample lightSample = sampleLight(vec3(k_nan), rvec_0);
    vec3 rndPnt = randomPointInSphere(rvec_x.xy);
    vec3 rndDir = normalize(lightSample.normal + rndPnt);
    Ray ray;
    initialize(ray, lightSample.position, rndDir);
    // Generate first vertex on light subpath and start random walk
    vec3 L = emission(lightSample.lightID,
            rndDir,
            0, lightSample);
    float p_pos = pdf_point_on_light(vec3(k_nan), lightSample);
    float p_dir = max(dot(rndDir, lightSample.normal), 0) / k_pi;
    float p = p_pos * p_dir;

    path.x[0].beta = L / p * abs_dot(lightSample.normal, rndDir);
    path.x[0].isect.position = ray.origin;
    path.x[0].isect.geometric_normal = lightSample.normal;
    path.x[0].pdfFwd = p_pos;
    path.x[0].pdfRev = 0;
    path.x[0].type = enum_vertex_type_light;

    path.n++;
    // Start random walk
    pathRandomWalk(path, ray, maxDepth-1, sample_stream, offset_light_subpath+1, p_dir, true);
    return path.n;
}

vec3 L(
    in const int    depth,
    in const ivec2  frameDim,
    out      ivec2  pRaster,
    in const vec4   sample_streams[num_states_vec4]
) {
    // Get the random vector for bdpt strategy
    // x: used to determine screen space pRaster
    // y: used to determine screen space pRaster
    // z: used to determine strategy
    const vec4 rvec0 = sample_streams[offset_camera_subpath + 0];
    const vec4 rvec1 = sample_streams[offset_strategy_selection + 0];
    // Determine the number of available strategies and pick a specific one
    int s, t, nStrategies;
    if (depth == 0) {
        nStrategies = 1;
        s = 0;
        t = 2;
    } else {
        nStrategies = depth + 2;
        s = min(int(rvec0.z * nStrategies), nStrategies - 1);
        t = nStrategies - s;
    }

    // Generate bdpt paths
    BDPTPath cameraSubpath;
    BDPTPath lightSubpath;

    // Generate a camera subpath with exactly t vertices
    vec2 pixel;
    pixel.x = rvec0.x * frameDim.x - 0.5;
    pixel.y = rvec0.y * frameDim.y - 0.5;

    if (generateCameraSubpath(t, pixel, frameDim, cameraSubpath, sample_streams) != t)
        return vec3(0);

    // Generate a light subpath with exactly s vertices
    if (generateLightSubpath(s, lightSubpath, sample_streams) != s)
        return vec3(0);

    // Execute connection strategy and return the radiance estimate
    vec2 newuv;
    float misWeight = 0.f;
    vec3 Lpath = connectBDPT(lightSubpath, cameraSubpath, s, t, newuv, misWeight, rvec1, frameDim);

    if (t == 1)
        pRaster = uv2pixel(newuv, frameDim);
    else {
        pixel.x = min(int(rvec0.x * frameDim.x), frameDim.x - 1);
        pixel.y = min(int(rvec0.y * frameDim.y), frameDim.y - 1);
        pRaster = ivec2(pixel);
    }
    
    return Lpath * nStrategies;
}

#endif