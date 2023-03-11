#ifndef _SRENDERER_SBDPT_COMMON_HEADER_
#define _SRENDERER_SBDPT_COMMON_HEADER_

#include "../include/common_trace.h"
#include "../include/common_rgen.h"

#define MIN_PATH_LENGTH 3 // avoid sampling direct illumination
#define MAX_PATH_LENGTH 6

#define MAX_EVENTS (MAX_PATH_LENGTH + 1)

/**
* @param beta: contains the product of the BSDF or phase function values, transmittances, and cosine terms
*              for the vertices in the path generated so far, divided by their respective sampling PDFs.
*              For the light subpath, they also include the emitted radiance divided by the density of the 
*              emission position and direction. For the camera subpath, radiance is replaced by importance.
* @param type: type of intersection:
*               - 0. camera
*               - 1. light
*               - 2. surface
*               - 3. medium
*/
struct BDPTVert {
    vec3 beta;  
    uint type;
    SurfaceIntersection isect;
    float pdfFwd;   // recording the probability per unit solid angle
    float pdfRev;   // the hypothetical probability density of the vertex if the direction of light transport was reversed
};

const uint enum_vertex_type_camera  = 0;
const uint enum_vertex_type_light   = 1;
const uint enum_vertex_type_surface = 2;
const uint enum_vertex_type_medium  = 3;

struct BDPTPath {
    BDPTVert x[MAX_EVENTS];
    int n;      // number of path vertices
};

void initialize(inout BDPTPath path) {
    path.n = 0;
}

bool isBlack(vec3 color) {
    return (color.x==0) && (color.y==0) && (color.z==0);
}

/**
* >> SRenderer :
* @param ipdf: possibility that select next from curr, in <solid angle measure>.
* @param curr: the vertex that is start from.
* @param next: the vertex that is selected.
* @ret   pdf : possibility that select next from curr, in <area measure>.
*/
float convertDensity(
    in const float ipdf, 
    in const BDPTVert curr,
    in const BDPTVert next
) {
    // TODO :: Return solid angle density if _next_ is an infinite area light
    // if (next.IsInfiniteLight())
    //     return pdf;
    float pdf = ipdf;
    const vec3 w = next.isect.position - curr.isect.position;
    const float dist2 = length_square(w);
    if (dist2 == 0) return 0;
    const float inv_dist2 = 1. / dist2;
    // if (next.IsOnSurface())
    pdf *= abs_dot(next.isect.geometric_normal, w * sqrt(inv_dist2));
    return pdf * inv_dist2;
}

/**
* >> SRenderer :
* @param curr: a vertex expected to be a light source.
* @param next: the vertex that is connected to.
* @ret   pdf : the posibility that the curr light select the direction that connect to next, in [area measurement].
*
* >> PBRT-v4 :
* pdf_light() computes the direction from the current vertex to the provided vertex
* and invokes Light::Pdf_Le() to retrieve the solid angle density of the underlying sampling strategy,
* which is subsequently converted into a density per unit area at v.
* In contrast to Vertex::Pdf(), this method also treats surface vertices located on area lights as if they were light source vertices.
*/
float pdf_light(
    in const BDPTVert curr,
    in const BDPTVert next
) {
    vec3 w = next.isect.position - curr.isect.position;
    float invDist2 = 1. / length_square(w);
    w *= sqrt(invDist2);    // normalize w
    float pdf;
    // TODO :: Compute sampling density for non-infinite light sources
    // We only use diffuse area light, so pdfDir is given by cosine weighted hemisphere sampling.
    float pdfDir = max(dot(curr.isect.geometric_normal, w), 0.) / k_pi;
    // Change to area measure
    pdf = pdfDir * invDist2;
    pdf *= abs_dot(next.isect.geometric_normal, w);
    return pdf;
}

/**
* >> SRenderer :
* @param curr: a vertex.
* @param prev: the vertex that is connected from.
* @param next: the vertex that is connected to.
* @ret   pdf : the posibility that the ? in [area measurement].
*
* >> PBRT-v4 :
*/
float pdf(
    in const BDPTVert curr,
    in const BDPTVert prev,
    in const BDPTVert next
) {
    if (curr.type == enum_vertex_type_light) {
        return pdf_light(curr, next);
    }
    // Compute directions to preceding and next vertex
    vec3 wp = normalize(prev.isect.position - curr.isect.position);
    vec3 wn = normalize(next.isect.position - curr.isect.position);
    // Compute directional density depending on the vertex type
    float pdf, unused;
    if (curr.type == enum_vertex_type_camera) {
        const float cosTheta = abs_dot(wn, curr.isect.geometric_normal);
        pdf = 1. / (globalUniform.cameraData.rectArea * cosTheta * cosTheta * cosTheta);
    }
    else if (curr.type == enum_vertex_type_surface)
        pdf = bsdf_pdf(curr.isect, wp, wn);
    // TODO :: light / medium cases
    // else if (curr.type == enum_vertex_type_medium)
    
    // Return probability per unit area at vertex next
    return convertDensity(pdf, curr, next);
}

/** helper function remap0 that deals with Dirac delta functions */
float remap0(float f) {
    return f != 0 ? f : 1;
}

/**
* >> SRenderer :
* @param curr: a vertex expected to be a light source.
* @param prev: the vertex that is connected to.
* @ret   pdf : the posibility that the curr light position is selected given the prev, in [area measurement].
* @ref   pbrt-v4 : PdfLightOrigin().
*
* >> PBRT-v4 :
* Note that the pdf() and pdf_light() methods use the directional probability density of the importance strategy
* implemented at the current vertex as measured at the location of another given vertex.
* However, this is not enough to fully characterize the behavior of path endpoints,
* whose sampling routines generate rays from a 4D distribution.
* An additional pdf_light_origin() method fills the gap by providing information about the spatial distribution
* of samples on the light sources themselves. For the same reason as before,
* a dedicated pdf_camera_origin() method for camera endpoints is not needed.
*/
float pdf_light_origin(
    in const BDPTVert curr,
    in const BDPTVert prev
) {
    if(!is_light(curr.isect)) return 0;
    // TODO :: Return solid angle density for non-infinite light sources

    // Get the probability of choosing the position
    LightSample lightSample;
    lightSample.position = curr.isect.position;
    lightSample.normal = curr.isect.geometric_normal;
    const float pdf_light = pdf_point_on_light(vec3(k_nan), lightSample);
    return pdf_light;   // is already pdfPos * pdfChoice
}

/**
* >> SRenderer :
* Given a specific connection strategy (s,t), return its MIS weight.
* MISWeight() considers other strategies that could have produced the same path.
* @ret:     balance heuristic MIS weight.
* @warn:    the t=0 case is omitted for simplicity
*/
float MISWeight(
    in BDPTPath lightPath,
    in BDPTPath cameraPath,
    in const BDPTVert sampled,
    in const int s, // light path vertex num
    in const int t  // camera path vertex num
) {
    if (s + t == 2) return 1;   // only 1 possible case for s+t=2, i.e. [s=0, t=2]

    // Temporarily update vertex properties for current strategy
    // So we store the vertices first and restore them later.
    BDPTVert qs;        if(s > 0) qs = lightPath.x[s - 1];
    BDPTVert pt;        if(t > 0) pt = cameraPath.x[t - 1];
    BDPTVert qsMinus;   if(s > 1) qsMinus = lightPath.x[s - 2];
    BDPTVert ptMinus;   if(t > 1) ptMinus = cameraPath.x[t - 2];

    {   // Look up connection vertices and their predecessors
        // Update sampled vertex for s=1 or t=1 strategy
        if (s == 1) lightPath.x[s - 1] = sampled;
        else if (t == 1) cameraPath.x[t - 1] = sampled;

        // Mark connection vertices as non-degenerate
        // ScopedAssignment<bool> a2, a3;
        // if (pt) a2 = { &pt->delta, false };
        // if (qs) a3 = { &qs->delta, false };

        // Update reverse density of vertex p_{t-1} and p_{t-2}
        if(t > 0) {
            cameraPath.x[t - 1].pdfRev =
                s > 0 ? pdf(lightPath.x[s - 1], 
                            ((s>1) ? lightPath.x[s - 2] : qsMinus),
                            cameraPath.x[t - 1])
                      : pdf_light_origin(cameraPath.x[t - 1], cameraPath.x[t - 2]);
        }
        if(t > 1) {
            cameraPath.x[t - 2].pdfRev =
                s > 0 ? pdf(cameraPath.x[t - 1],
                            lightPath.x[s - 1],
                            cameraPath.x[t - 2])
                      : pdf_light(cameraPath.x[t - 1], cameraPath.x[t - 2]);
        }

        // Update reverse density of vertices q_{s-2} and q_{s-2}
        if(s > 0) {
            lightPath.x[s - 1].pdfRev = pdf(
                cameraPath.x[t - 1], 
                ((t>1) ? cameraPath.x[t - 2] : ptMinus),
                lightPath.x[s - 1]);
        }
        if(s > 1) {
            lightPath.x[s - 2].pdfRev = pdf(
                lightPath.x[s - 1], 
                cameraPath.x[t - 1], 
                lightPath.x[s - 2]);
        }
    }
    // Accumulate MIS by iterating through the vertices
    float sumRi = 0;
    {   // Consider hypothetical connection strategies along the camera subpath
        float ri = 1;
        for (int i = t - 1; i > 0; --i) {
            ri *= remap0(cameraPath.x[i].pdfRev) /
                  remap0(cameraPath.x[i].pdfFwd);
            sumRi += ri;
        }
    }
    {   // Consider hypothetical connection strategies along the light subpath
        float ri = 1;
        for (int i = s - 1; i >= 0; --i) {
            ri *= remap0(lightPath.x[i].pdfRev) /
                  remap0(lightPath.x[i].pdfFwd);
            // A special case arises when the current strategy would involve intersecting a
            // light source (i.e., when ): this will fail when the endpoint involves
            // a Dirac delta distribution, hence the additional test below:
            sumRi += ri;
        }
    }
    // restore all the vertices
    if(s > 0) lightPath.x[s - 1]  = qs;
    if(t > 0) cameraPath.x[t - 1] = pt;
    if(s > 1) lightPath.x[s - 2]  = qsMinus;
    if(t > 1) cameraPath.x[t - 2] = ptMinus;
    
    return 1 / (1 + sumRi);
}

float correctShadingNormal_left(
    in const SurfaceIntersection isect,
    in const vec3 w
) {
    const float denom = abs_dot(w, isect.geometric_normal);
    if(denom==0) return 1;
    return abs_dot(w, isect.shading_frame[2]) / denom;
}

float correctShadingNormal_right(
    in const SurfaceIntersection isect,
    in const vec3 w
) {
    const float denom = abs_dot(w, isect.shading_frame[2]);
    if(denom==0) return 1;
    return abs_dot(w, isect.geometric_normal) / denom;
}

/**
* pathRandomWalk() traces paths starting at an initial vertex.
* Givens a start ray, it will takes care of sampling the following vertices and initializing the path array.
* @ret: the number of bounces in random walk.
*/
int pathRandomWalk(
    inout BDPTPath  path,
    in Ray ray,
    in const int maxDepth,
    inout uint RNG,
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

        // // Return when enough bounces has been traced
    	// if (++bounces >= maxDepth) {
        //     ++path.n;
        //     break;
        // }

        // Sample BSDF at current vertex and compute reverse probability
        vec2 bsdf_uv = vec2(stepAndOutputRNGFloat(RNG), stepAndOutputRNGFloat(RNG));
        vec3 dir_bsdf = bsdf_sample(
            path.x[path.n].isect,
            dir_view,
            bsdf_uv,
            stepAndOutputRNGFloat(RNG));

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
* @ret: return the number of vertices in the subpath.
*/
int generateCameraSubpath(
    in  int   maxCameraEvents,
    in  vec2  pixel,
    in  ivec2 frameDim,
    out BDPTPath path,
    inout uint RNG
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
    pathRandomWalk(path, ray, maxCameraEvents-1, RNG, pdf_dir, false);
    return path.n;
}

/**
* Generate light subpath.
* @ret: return the number of vertices in the subpath.
*/
int generateLightSubpath(
    in  int maxDepth,
    out BDPTPath path,
    inout uint RNG
) {
    initialize(path);
    // No path to build
    if (maxDepth == 0) return 0;
    // Sample initial ray for light subpath
    vec4 rnds = vec4(stepAndOutputRNGFloat(RNG),stepAndOutputRNGFloat(RNG),stepAndOutputRNGFloat(RNG),stepAndOutputRNGFloat(RNG));
    LightSample lightSample = sampleLight(vec3(k_nan), rnds);
    vec3 rndPnt = randomPointInSphere(RNG);
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
    pathRandomWalk(path, ray, maxDepth-1, RNG, p_dir, true);
    return path.n;
}

/**
* @param s: the number of vertices to use from the light subpath.
* @param t: the number of vertices to use from the camera subpath.
*
* ******** 0-index problem ****************
* An index of 0 means that no scattering events are used from the corresponding subpath.
* Only s=0 is supported, which furthermore requires cameraPath.x[t] to be a surface intersection involving a light source.
* Otherwise t=0 is not supported, light subpath could never hit a pinhole camera,
* so the the loop over camera subpaths starts at t=1.
*/

/**  Evaluates the portion of the measurement equation associated with a vertex. */
vec3 f(
    in const BDPTVert start,
    in const BDPTVert next
) {
    const vec3 wi = normalize(next.isect.position - start.isect.position);
    return bsdf_eval(start.isect, wi);
}

/** Compute the generalized geometry term */
vec3 G(
    in const BDPTVert v0,
    in const BDPTVert v1
) {
    vec3 d = v0.isect.position - v1.isect.position;
    float g = 1. / length_square(d);
    d *= sqrt(g);
    // g *= abs_dot(v0.isect.geometric_normal, d);
    // g *= abs_dot(v1.isect.geometric_normal, d);
    return g * transmittance(v0.isect, v1.isect);
}

vec3 Le(
    in const BDPTVert source,
    in const BDPTVert prev
) {
    if(!is_light(source.isect)) return vec3(0.);
    const vec3 w = normalize(prev.isect.position - source.isect.position);
    if (dot(source.isect.lightNormal, w) <= 0)
        return vec3(0.f);
    return lights[source.isect.lightID].intensity;
}

vec3 connectBDPT(
    in const BDPTPath lightPath,    // light path
    in const BDPTPath cameraPath,   // camera path
    in const int s,         // num of light path
    in const int t,         // num of camera path
    out vec2    newpixel,    // new pixel is selected if t=1 happens
    out float   out_misWeight,
    in const vec4 rnds,
    in const ivec2 resolution
) {
    vec3 L = vec3(0.);
    BDPTVert sampled;
    // Perform connection and write contribution to L
    if (s==0) {
        // Interpret the camera subpath as a complete path
        if(is_light(cameraPath.x[t - 1].isect)) {
            L = Le(cameraPath.x[t - 1], cameraPath.x[t - 2]) * cameraPath.x[t - 1].beta;
        }
    }
    else if (t==1) {
        // Sample a point on the camera and connect it to the light subpath
        vec4 proj = globalUniform.cameraData.viewProjMat * vec4(lightPath.x[s - 1].isect.position, 1.0);
        proj.xyz /= proj.w;
        newpixel =(proj.xy + vec2(1)) * 0.5;
        if(proj.z > 0 && proj.z < 1) {
            vec3 visible = transmittance(cameraPath.x[0].isect, lightPath.x[s - 1].isect);

            // Uniformly sample a lens interaction lensIntr
            vec3 lens_pos = globalUniform.cameraData.posW;
            vec3 lens_normal = normalize(globalUniform.cameraData.cameraW);
            vec3 wi = lens_pos - lightPath.x[s - 1].isect.position;
            float dist = length(wi);
            wi /= dist;
            float cosTheta = abs_dot(lens_normal, wi);
            float cos2Theta = cosTheta * cosTheta;

            float pdf = (dist * dist) / cosTheta;
            float lw = length(globalUniform.cameraData.cameraW);
            float lu = length(globalUniform.cameraData.cameraU) / lw;
            float lv = length(globalUniform.cameraData.cameraV) / lw;
            float A = lu * lv * 4;
            float we = 1. / (A * 1 * cos2Theta * cos2Theta);

            if (pdf > 0 && we!=0) {
                // Initialize dynamically sampled vertex and L for  case
                sampled.isect.position = lens_pos;
                sampled.isect.geometric_normal = lens_normal;
                sampled.beta = vec3(we / pdf);
                sampled.pdfFwd = 0;
                sampled.pdfRev = 0;
                sampled.type = enum_vertex_type_camera;

                L = lightPath.x[s - 1].beta
                    * bsdf_eval(lightPath.x[s - 1].isect, wi)
                    * correctShadingNormal_right(lightPath.x[s - 1].isect, wi)
                    * we / pdf
                    * visible;
            }
        }
    }
    else if (s==1) {
        // Sample a point on a light and connect it to the camera subpath
        // const Vertex &pt = cameraPath.x[t-1];
        LightSample lightSample = sampleLight(vec3(k_nan), rnds);
        float pdf = pdf_point_on_light(vec3(k_nan), lightSample);
        lightSample.rayDirection = normalize(lightSample.unoffsetPosition - cameraPath.x[t-1].isect.position);

        if(cameraPath.x[t-1].isect.lightID !=  lightSample.lightID) {
            // Initialize dynamically sampled vertex and L
            sampled.isect.position = lightSample.unoffsetPosition;
            sampled.isect.geometric_normal = lightSample.normal;
            sampled.isect.lightNormal = lightSample.normal;
            sampled.isect.lightID = lightSample.lightID;

            vec3 visible = transmittance(cameraPath.x[t-1].isect, sampled.isect);

            float g = max(-dot(lightSample.rayDirection, lightSample.normal), 0.f) /
                distance_squared(lightSample.unoffsetPosition, cameraPath.x[t-1].isect.position);

            vec3 le = Le(sampled, cameraPath.x[t-1]);

            if (pdf > 0 && !isBlack(le)) {
                L = le / pdf
                    * bsdf_eval(cameraPath.x[t-1].isect, normalize(lightSample.position - cameraPath.x[t-1].isect.position))
                    * cameraPath.x[t-1].beta
                    * visible
                    * g;
                
                sampled.beta = le / pdf;
                sampled.pdfFwd = pdf_light_origin(sampled, cameraPath.x[t-1]);
                sampled.pdfRev = 0;
                sampled.type = enum_vertex_type_light;
            }
        }
    }
    else {
        // Handle all other bidirectional connection cases
        vec3 wi = normalize(cameraPath.x[t - 1].isect.position - lightPath.x[s - 1].isect.position);
        L = lightPath.x[s - 1].beta
          * f(lightPath.x[s - 1], cameraPath.x[t - 1])
          * f(cameraPath.x[t - 1], lightPath.x[s - 1])
          * correctShadingNormal_right(lightPath.x[s - 1].isect, wi)
          * cameraPath.x[t - 1].beta;
        if (!isBlack(L))
            L *= G(lightPath.x[s - 1], cameraPath.x[t - 1]);
    }

    // Compute MIS weight for connection strategy
    // float misWeight = (s+t==2) ? 1 : 1./3;
    float misWeight = isBlack(L) ? 0.f : MISWeight(lightPath, cameraPath, sampled, s, t);
    L *= misWeight;
    out_misWeight = misWeight;

    return L;
}

#endif