#include "common/random.hlsli"
#include "common/sampling.hlsli"
#include "common/camera.hlsli"
#include "common/ltc.hlsli"
#include "srenderer/rayquery.hlsli"
#include "srenderer/materials/lambertian.hlsli"
#include "srenderer/shapes/rectangle.hlsli"

[[vk::push_constant]] cbuffer PushConstants { 
    int random_seed;
    int camera_index;
    int nee_mode;
    int estimator_mode;
    int accum_frame;
};

RWTexture2D<float4> u_image;
RWTexture2D<float4> u_image_cv;

struct RectangleLight {
    Rectangle shape;
    float3 emission;
};

struct scenehit_out {
    GeometryHit hit;
    int nearest_primitve;
    float nearest_distance;
};

void scene_hit_procedure<let N:int>(
    Array<RectangleLight, N> lights, Ray ray, 
    out float nearest_distance, out int nearest_primitve) {
    nearest_distance = 1e20;
    nearest_primitve = -1;
    PrimaryPayload shape_payload;
    for (int i = 0; i < N; ++i) {
        if (lights[i].shape.intersect(ray, shape_payload)) {
            const float dist = distance(shape_payload.hit.position, ray.origin);
            if (dist < nearest_distance) {
                nearest_primitve = i;
                nearest_distance = dist;
            }
        }
    }
}

scenehit_out scene_hit<let N : int>(Array<RectangleLight, N> lights, Ray ray) {
    PrimaryPayload payload = PrimaryRayQuery(ray);
    float prim_distance; int primitive;
    scene_hit_procedure(lights, ray, prim_distance, primitive);

    scenehit_out out;
    out.hit = payload.hit;
    out.nearest_distance = prim_distance;
    out.nearest_primitve = primitive;

    if ((HasHit(out.hit) && prim_distance > distance(payload.hit.position, ray.origin))) {
        out.nearest_primitve = -1;
        out.nearest_distance = distance(payload.hit.position, ray.origin);
    }
    return out;
}

Array<float, N> ltc_eval_lights<let N : int>(
    Array<RectangleLight, N> lights,
    float3 n, // normal
    float3 v, // view
    float3 p, // position
) {
    Array<float, N> result;
    for (int i = 0; i < N; ++i) {
        float3 diffuse = ltc::evaluate(
            n, v, p,
            identity3x3(),
            lights[i].shape.corners(), true);
        result[i] = diffuse.x;
    }
    return result;
}

float3 next_event_estimation<let N : int>(
    Array<RectangleLight, N> lights,
    int light_index,
    scenehit_out sh_o,
    float3 wi,
    LambertMaterial material,
    inout RandomSamplerState RNG,
) {
    ishape::sample_in i;
    i.uv = GetNextRandomFloat2(RNG);
    i.position = sh_o.hit.position;
    i.normal = sh_o.hit.geometryNormal;
    i.flag = nee_mode;
    
    ishape::sample sample_o = lights[light_index].shape.sample(i);
    float3 direction = normalize(sample_o.position - sh_o.hit.position);
    Ray ray = SpawnRay(sh_o.hit, direction);
    // evaluate the brdf
    ibsdf::eval_in eval_i;
    eval_i.wi = wi;
    eval_i.wo = direction;
    eval_i.geometric_normal = sh_o.hit.geometryNormal;
    eval_i.shading_frame = Frame(createFrame(sh_o.hit.shadingNormal));
    // evaluate the brdf
    float3 bsdf = LambertianBRDF::eval(eval_i, material);
    float3 throughput = bsdf / sample_o.pdf;
    // add direct lighting
    float3 radiance = float3(0);
    scenehit_out sh_o_2nd = scene_hit(lights, ray);
    if (sh_o_2nd.nearest_primitve == light_index) {
        radiance += throughput * lights[sh_o_2nd.nearest_primitve].emission;
    }
    return radiance;
}

float3 next_event_estimation_with_cv<let N : int>(
    Array<RectangleLight, N> lights,
    int light_index,
    scenehit_out sh_o,
    float3 wi,
    LambertMaterial material,
    inout RandomSamplerState RNG,
    inout float3 control_variate
) {
    ishape::sample_in i;
    i.uv = GetNextRandomFloat2(RNG);
    i.position = sh_o.hit.position;
    i.normal = sh_o.hit.geometryNormal;
    i.flag = nee_mode;

    ishape::sample sample_o = lights[light_index].shape.sample(i);
    float3 direction = normalize(sample_o.position - sh_o.hit.position);
    Ray ray = SpawnRay(sh_o.hit, direction);
    // evaluate the brdf
    ibsdf::eval_in eval_i;
    eval_i.wi = wi;
    eval_i.wo = direction;
    eval_i.geometric_normal = sh_o.hit.geometryNormal;
    eval_i.shading_frame = Frame(createFrame(sh_o.hit.shadingNormal));
    // evaluate the brdf
    float3 bsdf = LambertianBRDF::eval(eval_i, material);
    float3 throughput = bsdf / sample_o.pdf;
    // add direct lighting
    float3 radiance = float3(0);
    scenehit_out sh_o_2nd = scene_hit(lights, ray);
    if (sh_o_2nd.nearest_primitve == light_index) {
        radiance += throughput * lights[sh_o_2nd.nearest_primitve].emission;
    }
    control_variate = throughput * lights[light_index].emission;
    return radiance;
}

int sample_pdf_vec3(float3 pmf, float u, out float pdf) {
    // normalize pmf
    pmf /= pmf.x + pmf.y + pmf.z;
    if (u < pmf.x) {
        pdf = pmf.x;
        return 0;
    }
    if (u < pmf.x + pmf.y) {
        pdf = pmf.y;
        return 1;
    }
    pdf = pmf.z;
    return 2;
}

[shader("compute")]
[numthreads(32, 4, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(1024, 1024);
    const int2 pixel = int2(dtid.xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(dtid.xy, random_seed);

    
    float3 radiance = float3(0);
    float3 cv = float3(0);

    const float2 jitter = GetNextRandomFloat2(RNG) - 0.5f;
    Ray ray = generateRay(pixel + jitter, resolution, false, GPUScene_camera[camera_index]);

    Array<RectangleLight, 3> lights = {
        { Rectangle(float3(4, 10, 0), float3(0, 1.57, 0), 5, 5), float3(5.0f, 0.f, 0.f) },
        { Rectangle(float3(4, 8, -6), float3(0, 2, 0), 5, 5), float3(0.0f, 5.f, 0.f) },
        // { Rectangle(float3(4, 8, 6), float3(0, -2, 0), 5, 5), float3(0.0f, 0.f, 5.f) },
        { Rectangle(float3(4, 8, 6), float3(0, -2, 0), 5, 5), float3(0.0f, 0.f, 5.f) },
        // { Rectangle(float3(4, 10, 0), float3(0, 1.57, 0), 5, 5), float3(5.0f, 1.f, 2.f) },
        // { Rectangle(float3(4, 8, -6), float3(0, 2, 0), 5, 5), float3(2.0f, 5.f, 1.f) },
        // // { Rectangle(float3(4, 8, 6), float3(0, -2, 0), 5, 5), float3(0.0f, 0.f, 5.f) },
        // { Rectangle(float3(4, 8, 5), float3(0, -2, 0), 5, 5), float3(1.0f, 4.f, 5.f) },
    };

    scenehit_out sh_o = scene_hit(lights, ray);

    // If the ray directly hits a light, return the light's emission

    if (sh_o.nearest_primitve != -1) {
        radiance = lights[sh_o.nearest_primitve].emission;
        if (accum_frame == 0) {
            u_image[pixel] = float4(radiance, 1.0f);
            u_image_cv[pixel] = float4(float3(0), 1.0f);
        }
        else if (accum_frame == 1) {
            u_image[pixel] += float4(radiance, 1.0f);
            u_image_cv[pixel] += float4(float3(0), 1.0f);
        }
        return;
    }

    // else evaluate the direct lighting
    LambertMaterial material = LambertMaterial();
    material.R = float3(1.f);
    LambertianBRDF brdf = LambertianBRDF();
    Ray primary_ray = ray;
    if (HasHit(sh_o.hit)) {

        Array<float, 3> ltc_evals = ltc_eval_lights(
            lights, sh_o.hit.geometryNormal,
            -ray.direction, sh_o.hit.position);
        
        [ForceUnroll]
        for (int i = 0; i < 3; ++i) {
            // reset the ray and throughput
            float3 throughput = float3(1.0f);
            ray = primary_ray;

            // brdf importance sampling
            if (estimator_mode == 0) {
                // sample the brdf
                ibsdf::sample_in i;
                i.u = GetNextRandomFloat3(RNG);
                i.wi = -ray.direction;
                i.geometric_normal = sh_o.hit.geometryNormal;
                i.shading_frame = Frame(createFrame(sh_o.hit.shadingNormal));
                ibsdf::sample_out sample_o = brdf::sample(i, material);
                ray = SpawnRay(sh_o.hit, sample_o.wo);
                throughput *= sample_o.bsdf;
                // add direct lighting
                scenehit_out sh_o_2nd = scene_hit(lights, ray);
                if (sh_o_2nd.nearest_primitve != -1) {
                    radiance += throughput * lights[sh_o_2nd.nearest_primitve].emission;
                }
            }
            // LTC closed form solution
            else if (estimator_mode == 1) {
                for (int i = 0; i < 3; ++i) {
                    radiance += ltc_evals[i] * lights[i].emission;
                }
            }
            // luminance light sampling
            else if (estimator_mode == 2) {
                const float3 ray_in = -ray.direction;
                // sample the light shape
                float3 pmf = {
                    luminance(lights[0].emission * ltc_evals[0]),
                    luminance(lights[1].emission * ltc_evals[1]),
                    luminance(lights[2].emission * ltc_evals[2])
                };

                float pdf;
                int random_index = sample_pdf_vec3(pmf, GetNextRandom(RNG), pdf);

                float3 radiance_nee = next_event_estimation(
                    lights, random_index, sh_o, ray_in, material, RNG);
                radiance += radiance_nee / pdf;
            }
            // moment2 light sampling
            else if (estimator_mode == 3) {
                const float3 ray_in = -ray.direction;
                float3 pmf = {
                    length(lights[0].emission * ltc_evals[0]),
                    length(lights[1].emission * ltc_evals[1]),
                    length(lights[2].emission * ltc_evals[2])
                };

                float pdf;
                int random_index = sample_pdf_vec3(pmf, GetNextRandom(RNG), pdf);

                float3 radiance_nee = next_event_estimation(
                    lights, random_index, sh_o, ray_in, material, RNG);
                radiance += radiance_nee / pdf;
            }
            // control variates
            else if (estimator_mode == 6) {
                const float3 ray_in = -ray.direction;
                // sample the light shape
                float3 pmf = {
                    luminance(lights[0].emission * ltc_evals[0]),
                    luminance(lights[1].emission * ltc_evals[1]),
                    luminance(lights[2].emission * ltc_evals[2])
                };

                float pdf;
                int random_index = sample_pdf_vec3(pmf, GetNextRandom(RNG), pdf);

                float3 radiance_cv = float3(0);
                float3 radiance_nee = next_event_estimation_with_cv(
                    lights, random_index, sh_o, ray_in, material, RNG, radiance_cv);
                radiance += (radiance_nee - radiance_cv) / pdf;
                // radiance += (radiance_cv) / pdf;
                
                for (int i = 0; i < 3; ++i) {
                    cv += ltc_evals[i] * lights[i].emission;
                }
            }
            else if (estimator_mode == 7) {
                const float3 ray_in = -ray.direction;
                // sample the light shape
                float3 pmf = {
                    luminance(lights[0].emission * ltc_evals[0]),
                    luminance(lights[1].emission * ltc_evals[1]),
                    luminance(lights[2].emission * ltc_evals[2])
                };

                float pdf;
                int random_index = sample_pdf_vec3(pmf, GetNextRandom(RNG), pdf);

                float3 radiance_cv = float3(0);
                float3 radiance_nee = next_event_estimation_with_cv(
                    lights, random_index, sh_o, ray_in, material, RNG, radiance_cv);
                radiance += radiance_nee / pdf;
                
                float3 cv_integrated = float3(0);
                for (int i = 0; i < 3; ++i) {
                    cv_integrated += ltc_evals[i] * lights[i].emission;
                }
                cv += radiance_cv / pdf / cv_integrated;
            }
        }
        radiance /= 3;
        cv /= 3;

        // channel-stratified sampling
        if (estimator_mode == 4) {
            const float3 ray_in = -ray.direction;
            const float3 pmf_r = {
                length(lights[0].emission.r * ltc_evals[0]),
                length(lights[1].emission.r * ltc_evals[1]),
                length(lights[2].emission.r * ltc_evals[2])
            };
            const float3 pmf_g = {
                length(lights[0].emission.g * ltc_evals[0]),
                length(lights[1].emission.g * ltc_evals[1]),
                length(lights[2].emission.g * ltc_evals[2])
            };
            const float3 pmf_b = {
                length(lights[0].emission.b * ltc_evals[0]),
                length(lights[1].emission.b * ltc_evals[1]),
                length(lights[2].emission.b * ltc_evals[2])
            };

            float pdf_r; float pdf_g; float pdf_b;
            int li_r = sample_pdf_vec3(pmf_r, GetNextRandom(RNG), pdf_r);
            int li_g = sample_pdf_vec3(pmf_g, GetNextRandom(RNG), pdf_g);
            int li_b = sample_pdf_vec3(pmf_b, GetNextRandom(RNG), pdf_b);

            float radiance_r = next_event_estimation(
                lights, li_r, sh_o, ray_in, material, RNG).r;
            float radiance_g = next_event_estimation(
                lights, li_g, sh_o, ray_in, material, RNG).g;
            float radiance_b = next_event_estimation(
                lights, li_b, sh_o, ray_in, material, RNG).b;
            radiance += float3(
                radiance_r / pdf_r,
                radiance_g / pdf_g,
                radiance_b / pdf_b
            );
        }
        else if (estimator_mode == 5) {
            const float3 ray_in = -ray.direction;
            float3 radiance_0 = next_event_estimation(
                lights, 0, sh_o, ray_in, material, RNG);
            float3 radiance_1 = next_event_estimation(
                lights, 1, sh_o, ray_in, material, RNG);
            float3 radiance_2 = next_event_estimation(
                lights, 2, sh_o, ray_in, material, RNG);
            radiance += radiance_0 + radiance_1 + radiance_2;
        }
    }

    if (accum_frame == 0) {
        u_image[pixel] = float4(radiance, 1.0f);
        u_image_cv[pixel] = float4(cv, 1.0f);
    }
    else if (accum_frame == 1) {
        u_image[pixel] += float4(radiance, 1.0f);
        u_image_cv[pixel] += float4(cv, 1.0f);
    }
}