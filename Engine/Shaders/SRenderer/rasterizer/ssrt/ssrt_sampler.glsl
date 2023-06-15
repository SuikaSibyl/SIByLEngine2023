#ifndef _HEADER_SSRT_SAMPLER_
#define _HEADER_SSRT_SAMPLER_

/*****************************************************
**              Interface Definition                **
*****************************************************/
struct InitStateTS {
    RayStartInfo info;
    vec3 normalInVS;
};

struct SampleTS {
    ivec2        xy;
    vec2         uv;
    vec3         normalInVS;
    float        pdf;
    RayStartInfo info;
    vec3         radiance;
    bool         hasIsect;
};

bool unpackVSInfo(
    in const uvec2 tid,
    out InitStateTS state
) {
    bool isect = unpackVSInfo(tid, state.info);
    state.normalInVS = state.info.sampleNormalInVS;
    return isect;
}

bool unpackVSInfo_Vec(
    in const vec2 uv,
    out InitStateTS state
) {
    bool isect = unpackVSInfo_vec(uv, state.info);
    state.normalInVS = state.info.sampleNormalInVS;
    return isect;
}
/*****************************************************
**                        Ray                       **
*****************************************************/

const uint EnumIsectResult_None = 0;
const uint EnumIsectResult_True = 1;
const uint EnumIsectResult_MismatchDir = 2;
const uint EnumIsectResult_NegativeCos = 3;
const uint EnumIsectResult_Err = 4;

uint FindIntersection(
    in const InitStateTS state,
    in const vec3 rayDirInVS,
    out SampleTS isect
) {
    const float NormalCheckThreshold = 1;
    const float MatchDirectionThreshold = 0.999;
    SSRTConfig config = SSRTConfig(false);

    SSRay ray = PrepareSSRT(state.info, rayDirInVS);
    const float NdL = dot(state.normalInVS, rayDirInVS);
    if(NdL < 0.) return EnumIsectResult_NegativeCos;
    else if(NdL < NormalCheckThreshold) config.normal_check = true;

    vec3 intersection;
    if(FindIntersection_Interface(ray, intersection, config)) {
        isect.xy = ivec2(intersection.xy * gUniform.view_size);
        isect.uv = intersection.xy;
        bool hasIsect = unpackVSInfo(isect.xy, isect.info);
        isect.normalInVS = isect.info.sampleNormalInVS;
        if(!hasIsect) {
            return EnumIsectResult_Err;
        }
        // const vec3 actualDir = normalize((isect.info.samplePosInVS - state.info.samplePosInVS).xyz);
        // if(dot(actualDir, rayDirInVS) < MatchDirectionThreshold) {
        //     return EnumIsectResult_MismatchDir;
        // }
        if(dot(isect.normalInVS, -rayDirInVS) < 0.) {
        // || dot(rayDirInVS, actualDir) < 0.) {
            return EnumIsectResult_NegativeCos;
        }
        return EnumIsectResult_True;
    }
    return EnumIsectResult_None;
}


uint FindOcclusion(
    in const InitStateTS beg_state,
    in const InitStateTS end_state,
    out SampleTS isect
) {
    const float NormalCheckThreshold = 1;
    const float MatchDirectionThreshold = 0.999;
    SSRTConfig config = SSRTConfig(false);
    
    vec3 rayDirInVS = end_state.info.samplePosInVS.xyz - beg_state.info.samplePosInVS.xyz;
    const float dist = length(rayDirInVS);
    rayDirInVS = normalize(rayDirInVS);

    SSRay ray = PrepareSSRT(beg_state.info, rayDirInVS);
    const float NdL = dot(beg_state.normalInVS, rayDirInVS);
    if(NdL < 0.) return EnumIsectResult_NegativeCos;
    else if(NdL < NormalCheckThreshold) config.normal_check = true;

    vec3 intersection;
    if(FindOcclusion_DDA(ray, end_state.info.samplePosInTS.xy, intersection, config)) {
        isect.xy = ivec2(intersection.xy * gUniform.view_size);
        isect.uv = intersection.xy;
        bool hasIsect = unpackVSInfo(isect.xy, isect.info);
        isect.normalInVS = isect.info.sampleNormalInVS;
        if(!hasIsect) {
            return EnumIsectResult_Err;
        }
        // const vec3 actualDir = normalize((isect.info.samplePosInVS - state.info.samplePosInVS).xyz);
        // if(dot(actualDir, rayDirInVS) < MatchDirectionThreshold) {
        //     return EnumIsectResult_MismatchDir;
        // }
        if(dot(isect.normalInVS, -rayDirInVS) < 0.
        || dot(-rayDirInVS, vec3(0,0,-1)) < 0.) {
        // || dot(rayDirInVS, actualDir) < 0.) {
            return EnumIsectResult_NegativeCos;
        }
        return EnumIsectResult_True;
    }
    return EnumIsectResult_None;
}

/*****************************************************
**      SampleTech Cosine Weighted Sampling         **
*****************************************************/
SampleTS SampleTech_CosWeight(
    in const InitStateTS state,
    inout uint RNG
) {
    const vec3 dirInVS = normalize(normalize(state.normalInVS) + randomPointInSphere(RNG));
    SampleTS isect;
    isect.pdf = 0;
    uint result = FindIntersection(state, dirInVS, isect);
    isect.hasIsect = false;
    isect.radiance = vec3(0);
    if(result == EnumIsectResult_True) {
        isect.hasIsect = true;
        isect.radiance = texelFetch(di, isect.xy, 0).xyz * k_pi;
        const float dist = distance(isect.info.samplePosInVS.xyz, state.info.samplePosInVS.xyz);
        isect.pdf = max(dot(state.normalInVS, dirInVS), 0.)
            * max(dot(isect.normalInVS, -dirInVS), 0.)  / (k_pi * dot(dist,dist));
        if(isect.radiance == vec3(0) || isect.pdf == 0) {
            isect.hasIsect = false;
        }
    }
    return isect;
}

float SampleTechPdf_CosWeight(
    in const InitStateTS state,
    in const SampleTS samplets
) {
    const float dist = distance(samplets.info.samplePosInVS.xyz, state.info.samplePosInVS.xyz);
    const vec3 dirInVS = normalize(samplets.info.samplePosInVS.xyz - state.info.samplePosInVS.xyz);
    return  max(dot(state.normalInVS, dirInVS), 0.)
            * max(dot(samplets.normalInVS, -dirInVS), 0.)  / (k_pi * dot(dist,dist));
}

/*****************************************************
**          SampleTech Light Cut Sampling           **
*****************************************************/

vec2 four_neighbors[4] = vec2[](
    vec2(-1., +1.),
    vec2(+1., +1.),
    vec2(+1., -1.),
    vec2(-1., -1.)
);

ivec2 four_neighbors_pix[4] = ivec2[](
    ivec2(0, 1),
    ivec2(1, 1),
    ivec2(1, 0),
    ivec2(0, 0)
);

float min_distance(
    in const AABB aabb,
    in const vec3 pos
) {
    const vec3 d = aabb.max - aabb.min;
    const vec3 m = (aabb.max + aabb.min) * 0.5;
    const vec3 n = pos - m;
    vec3 f = abs(n) - d * 0.5;
    vec3 c = max(f, vec3(0.));
    return length(c);
}

float max_distance(
    in const AABB aabb,
    in const vec3 pos
) {
    vec3 dis_min = abs(pos - aabb.min);
    vec3 dis_max = abs(pos - aabb.max);
    vec3 dis = max(dis_max, dis_min);
    return length(dis);
}

float geometry_term(
    in const AABB aabb,
    in const NormalConeEx normalCone,
    in const vec3 pos,
    in const vec3 N
) {
    const vec3 m = (aabb.max + aabb.min) * 0.5;
    const vec3 dir = normalize(m - pos);
    float maximum_cos = 0.f;

    vec3 corner[8] = vec3[](
        vec3(aabb.min.x, aabb.min.y, aabb.min.z),
        vec3(aabb.max.x, aabb.min.y, aabb.min.z),
        vec3(aabb.min.x, aabb.max.y, aabb.min.z),
        vec3(aabb.min.x, aabb.min.y, aabb.max.z),
        vec3(aabb.max.x, aabb.max.y, aabb.min.z),
        vec3(aabb.max.x, aabb.min.y, aabb.max.z),
        vec3(aabb.min.x, aabb.max.y, aabb.max.z),
        vec3(aabb.max.x, aabb.max.y, aabb.max.z)
    );

    for(int i=0; i<8; ++i) {
        corner[i] = normalize(corner[i] - pos);
        maximum_cos = max(maximum_cos, dot(corner[i], dir));
    }

    const float theta = acos(dot(normalCone.direction, -dir));

    const float theta_u = acos(maximum_cos);
    const float theta_i = acos(dot(N, dir));
    const float theta_i_prim = max(theta_i - theta_u, 0);
    const float theta_prim = max(theta - normalCone.theta_o - theta_u, 0);

    if(theta_prim < normalCone.theta_e) {
        return cos(theta_prim) * abs(cos(theta_i_prim));
    }
    else {
        return 0.001;
    }
}

vec4 ComputeChildNodeProb(
    in const ivec2 xy,
    in const int mip_level,
    in const RayStartInfo startInfo,
    in const vec3 sampleNormalInVS
) {
    vec4 child_importance = vec4(0);
    if(gUniform.lightcut_mode == 0) {
        for (int j=0; j<4; ++j) {
            ivec2 xy_offset = four_neighbors_pix[j];
            float lumin_importance = texelFetch(importance_mip, xy + xy_offset, mip_level).x;
            child_importance[j] = lumin_importance;
        }
        const float sum = dot(vec4(1), child_importance);
        child_importance /= sum;
    }
    else if(gUniform.lightcut_mode >= 1 && gUniform.lightcut_mode <= 6) {
        vec4 lum_importance = vec4(0);
        vec4 min_importance = vec4(0);
        vec4 max_importance = vec4(0);

        int singluarity = 0;
        int singluarity_id = 0;

        for (int j=0; j<4; ++j) {
            ivec2 xy_offset = four_neighbors_pix[j];
            float lumin_importance = texelFetch(importance_mip, xy + xy_offset, mip_level).x;

            // LC Weight estimate
            vec4 bbxy = texelFetch(boundingbox_mip, xy + xy_offset, mip_level);
            vec4 bbnc = texelFetch(bbncpack_mip, xy + xy_offset, mip_level);
            const vec3 conedir = unpackNormal(texelFetch(normalcone_mip, xy + xy_offset, mip_level).xyz);
            AABB aabb = AABB(vec3(bbxy.xy, bbnc.x), vec3(bbxy.zw, bbnc.y));
            const NormalConeEx normal_cone = NormalConeEx(conedir, bbnc.z, bbnc.w, true);
            // geometry
            // distance
            float min_dist = min_distance(aabb, startInfo.samplePosInVS.xyz);
            float max_dist = max_distance(aabb, startInfo.samplePosInVS.xyz);
            float g_term = 1.f;
            if(gUniform.lightcut_mode == 2) {
                g_term = (min_dist == 0) // inside the aabb?
                    ? 1.f
                    : geometry_term(aabb, normal_cone, startInfo.samplePosInVS.xyz, sampleNormalInVS.xyz);
            }
            else if(gUniform.lightcut_mode == 3 || gUniform.lightcut_mode == 4) {
                if(dot(normal_cone.direction, sampleNormalInVS.xyz) > 0.999
                 && normal_cone.theta_o < 0.01) {
                    g_term = 0.f;
                }
            }
            else if(gUniform.lightcut_mode == 5) {
                const float cdn = dot(normal_cone.direction, sampleNormalInVS.xyz);
                if(cdn > 0.999 && normal_cone.theta_o < 0.01) {
                    g_term = 0.f;
                }
                else {
                    const float omcdn = max(1. - cdn, 0);
                    g_term = omcdn;
                }
            }
            lumin_importance *= g_term;
            if(min_dist == 0) {
                singluarity++;
                singluarity_id = j;
            }

            lum_importance[j] = lumin_importance;
            min_importance[j] = lumin_importance / (min_dist * min_dist);
            max_importance[j] = lumin_importance / (max_dist * max_dist);
        }

        if(gUniform.lightcut_mode == 3 || gUniform.lightcut_mode == 5) {
            float parent = dot(vec4(1), lum_importance);
            if(parent == 0) child_importance = vec4(0);
            else child_importance = lum_importance / parent;
        }
        else {
            if (singluarity == 1) {
                min_importance = vec4(0);
                min_importance[singluarity_id] = 1;
            }
            else if(singluarity > 1) {
                min_importance = lum_importance;
            }
            {
                float min_parent = dot(vec4(1), min_importance);
                float max_parent = dot(vec4(1), max_importance);
                child_importance = 0.5 * (min_importance / min_parent + max_importance / max_parent);
            }

            if(gUniform.lightcut_mode == 6) {
                float lum_parent = dot(vec4(1), lum_importance);
                lum_importance /= lum_parent;
                child_importance = lum_importance - 0.2 * child_importance;
                child_importance = max(child_importance, vec4(0.0001));
                float sum = dot(vec4(1), child_importance);
                child_importance /= sum;
            }
        }
    }
    else if(gUniform.lightcut_mode == 4) {
        vec4 lum_importance = vec4(0);
        vec4 min_importance = vec4(0);
        vec4 max_importance = vec4(0);

        int singluarity = 0;
        int singluarity_id = 0;

        for (int j=0; j<4; ++j) {
            ivec2 xy_offset = four_neighbors_pix[j];
            float lumin_importance = texelFetch(importance_mip, xy + xy_offset, mip_level).x;

            // LC Weight estimate
            vec4 bbxy = texelFetch(boundingbox_mip, xy + xy_offset, mip_level);
            vec4 bbnc = texelFetch(bbncpack_mip, xy + xy_offset, mip_level);
            const vec3 conedir = unpackNormal(texelFetch(normalcone_mip, xy + xy_offset, mip_level).xyz);
            AABB aabb = AABB(vec3(bbxy.xy, bbnc.x), vec3(bbxy.zw, bbnc.y));
            const NormalConeEx normal_cone = NormalConeEx(conedir, bbnc.z, bbnc.w, true);
            // geometry
            // distance
            float min_dist = min_distance(aabb, startInfo.samplePosInVS.xyz);
            float max_dist = max_distance(aabb, startInfo.samplePosInVS.xyz);
            float g_term = 1.f;
            if(gUniform.lightcut_mode == 2) {
                g_term = (min_dist == 0) // inside the aabb?
                    ? 1.f
                    : geometry_term(aabb, normal_cone, startInfo.samplePosInVS.xyz, sampleNormalInVS.xyz);
            }
            // float g_term = (min_dist == 0) // inside the aabb?
            //     ? 1.f
            //     : geometry_term(aabb, normal_cone, startInfo.samplePosInVS.xyz, sampleNormalInVS.xyz);
            lumin_importance *= g_term;
            if(min_dist == 0) {
                singluarity++;
                singluarity_id = j;
            }

            lum_importance[j] = lumin_importance;
            min_importance[j] = lumin_importance / (min_dist * min_dist);
            max_importance[j] = lumin_importance / (max_dist * max_dist);
        }
        
        if (singluarity == 1) {
            min_importance = vec4(0);
            min_importance[singluarity_id] = 1;
        }
        else if(singluarity > 1) {
            min_importance = lum_importance;
        }
        {
            float min_parent = dot(vec4(1), min_importance);
            float max_parent = dot(vec4(1), max_importance);
            child_importance = 0.5 * (min_importance / min_parent + max_importance / max_parent);
        }
    }
    return child_importance;
}

vec2 sampleImportanceUV_LC_v0(
    inout uint RNG,
    out float probability,
    in const RayStartInfo startInfo,
    in const vec3 sampleNormalInVS
) {
    // gImportanceMIP
    int mip_level = 11;
    vec2 uv = vec2(0.5, 0.5);
    ivec2 xy = ivec2(0);

    float p = 1.f;
    float pixel_size = 1. / 2;
    
    for (int i = 0; i < gUniform.is_depth; ++i) {
        pixel_size /= 2;    // half pixel size
        mip_level--; // next mip level
        xy *= 2; // sample next level

        // sample next level
        float rnd = UniformFloat(RNG);
        float accum = 0.f;
        int last_non_zero = -1;
        float last_non_zero_imp = 0;
        float last_non_zero_pdf = 0;

        vec4 child_importance = ComputeChildNodeProb(xy, mip_level, startInfo, sampleNormalInVS);

        for (int j=0; j<4; ++j) {
            vec2 uv_offset = four_neighbors[j] * pixel_size;
            ivec2 xy_offset = four_neighbors_pix[j];
            float importance = child_importance[j];
            float nimportance = importance;
            accum += nimportance;
            
            if(nimportance > 0) {
                last_non_zero = j;
                last_non_zero_pdf = nimportance;
                last_non_zero_imp = importance;
            }

            if(rnd < accum) {
                uv = uv + uv_offset;
                xy = xy + xy_offset;
                p = p * nimportance;
                break;
            }
            else if(j==3 && last_non_zero!=-1) {
                uv = uv + four_neighbors[last_non_zero] * pixel_size;
                xy = xy + four_neighbors_pix[last_non_zero];
                p = p * last_non_zero_pdf;
                break;
            }
            else {
                // should not happen...
            }
        }
        p *= 4;
    }

    vec2 uv_pertub = vec2(UniformFloat(RNG), UniformFloat(RNG)); // (0,1)
    uv_pertub = vec2(-1, -1) + uv_pertub * 2; // (-1, 1)
    uv += uv_pertub * pixel_size;
    probability = p;
    return uv;
}

vec2 sampleImportanceUV_LC_v0_v1(
    inout uint RNG,
    out float probability,
    in const RayStartInfo startInfo,
    in const vec3 sampleNormalInVS
) {
    // gImportanceMIP
    int mip_level = 11;
    vec2 uv = vec2(0.5, 0.5);
    ivec2 xy = ivec2(0);

    float p = 1.f;
    float pixel_size = 1. / 2;
    
    for (int i = 0; i < gUniform.is_depth; ++i) {
        pixel_size /= 2;    // half pixel size
        mip_level--; // next mip level
        xy *= 2; // sample next level

        // sample next level
        float rnd = UniformFloat(RNG);
        float accum = 0.f;
        int last_non_zero = -1;
        float last_non_zero_imp = 0;
        float last_non_zero_pdf = 0;

        vec4 child_importance = ComputeChildNodeProb(xy, mip_level, startInfo, sampleNormalInVS);

        for (int j=0; j<4; ++j) {
            vec2 uv_offset = four_neighbors[j] * pixel_size;
            ivec2 xy_offset = four_neighbors_pix[j];
            float importance = child_importance[j];
            float nimportance = importance;
            accum += nimportance;
            
            if(nimportance > 0) {
                last_non_zero = j;
                last_non_zero_pdf = nimportance;
                last_non_zero_imp = importance;
            }

            if(rnd < accum) {
                uv = uv + uv_offset;
                xy = xy + xy_offset;
                p = p * nimportance;
                break;
            }
            else if(j==3 && last_non_zero!=-1) {
                uv = uv + four_neighbors[last_non_zero] * pixel_size;
                xy = xy + four_neighbors_pix[last_non_zero];
                p = p * last_non_zero_pdf;
                break;
            }
            else {
                // should not happen...
            }
        }
    }

    for(int i=gUniform.is_depth; i<11; ++i) {
        p /= 4;
    }

    vec2 uv_pertub = vec2(UniformFloat(RNG), UniformFloat(RNG)); // (0,1)
    uv_pertub = vec2(-1, -1) + uv_pertub * 2; // (-1, 1)
    uv += uv_pertub * pixel_size;

    probability = p;
    return uv;
}

float sampleImportanceUV_pdf(
    in const vec2 sample_uv,
    in const RayStartInfo startInfo,
    in const vec3 sampleNormalInVS
) {
    // gImportanceMIP
    vec2 samp_uv = vec2(floor(sample_uv * 2048) + vec2(0.5)) / 2048;
    int mip_level = 11;
    vec2 uv = vec2(0.5, 0.5);
    ivec2 xy = ivec2(0);
    
    float p = 1.f;
    float pixel_size = 1. / 2;
    
    for (int i = 0; i < gUniform.is_depth; ++i) {
        pixel_size /= 2;    // half pixel size
        mip_level--; // next mip level
        xy *= 2; // sample next level

        vec4 child_importance = ComputeChildNodeProb(xy, mip_level, startInfo, sampleNormalInVS);

        vec2 uv_direction = samp_uv - uv;
        uv_direction.x = (uv_direction.x >= 0) ? 1. : -1.;
        uv_direction.y = (uv_direction.y >= 0) ? 1. : -1.;
        
        for (int j=0; j<4; ++j) {
            bool isChosen = (uv_direction == four_neighbors[j]);
            if(isChosen) {
                vec2 uv_offset = four_neighbors[j] * pixel_size;
                ivec2 xy_offset = four_neighbors_pix[j];
                float importance = child_importance[j];
                // if(isnan(importance)) importance = 0;
                uv = uv + uv_offset;
                xy = xy + xy_offset;
                p = p * importance;
                break;
            }
        }
        p *= 4;
    }
    return p;
}

float jacobian_as2ss(
    in const InitStateTS initState,
    in const SampleTS samplets,
    out float ss2ss_pdf
) {    
    const vec2 sample_uv = samplets.uv;
    vec3 pix_position = globalUniform.cameraData.cameraW
                + (-0.5 + sample_uv.x) * 2 * globalUniform.cameraData.cameraU
                + (-0.5 + sample_uv.y) * 2 * globalUniform.cameraData.cameraV;
    vec3 pix_dir = normalize(pix_position);
    float cos_0 = max(dot(-pix_dir, -normalize(globalUniform.cameraData.cameraW)), 0);
    float r_0 = length(pix_position);
    float area_0 = 4 * length(globalUniform.cameraData.cameraU) * length(globalUniform.cameraData.cameraV);
    float jacobian_0 = cos_0 * area_0 / (r_0 * r_0);

    float dist_1 = length(samplets.info.samplePosInVS.xyz);
    vec3 dir_1 = normalize(samplets.info.samplePosInVS.xyz);
    float cos_1 = max(dot(samplets.normalInVS, -dir_1), 0);
    float jacobian_1 = jacobian_0 * (dist_1 * dist_1) / cos_1;

    const vec3 reflectionDirVS = normalize(samplets.info.samplePosInVS.xyz - initState.info.samplePosInVS.xyz);
    float cos_2 = max(dot(samplets.normalInVS, -reflectionDirVS), 0);
    float dist_2 = max(length(samplets.info.samplePosInVS - initState.info.samplePosInVS), 0);

    if (dist_2 != 0 && !isinf(dist_2) && !isnan(dist_2) && !isinf(jacobian_1)) {
        float g = cos_2 / (dist_2 * dist_2);
        float jacobian_2 = jacobian_1 * g;
    
        float cos_3 = max(dot(reflectionDirVS, initState.normalInVS), 0);

        ss2ss_pdf = jacobian_2 * cos_3;
        return jacobian_1;
    }
    else {
        ss2ss_pdf = 0;
        return 0;
    }
}

vec3 texture_sample_jacobian(
    in const vec2 sample_uv,
    in const InitStateTS initState,
    out float area_prob
) {
    area_prob = 0;
    
    InitStateTS endState;
    bool hasIsect = unpackVSInfo_Vec(sample_uv, endState);

    if (!hasIsect) {
        return vec3(0);
    }

    // check visibility
    vec3 sample_color = vec3(0);
    vec3 reflectionDirVS = normalize(endState.info.samplePosInVS.xyz - initState.info.samplePosInVS.xyz);

    {
        SampleTS isect;
        uint result = FindOcclusion(initState, endState, isect);
        if(result == EnumIsectResult_None) {
            // ivec2 offsetXY = abs(isect.xy - ivec2(sample_uv * gUniform.view_size));
            // InitStateTS realEndState;
            // unpackVSInfo(isect.xy, realEndState);
            // if(offsetXY.x > 1 || offsetXY.y > 1) {
            //     return vec3(0);
            // }
        }
        else {
            return vec3(0);
        }
    }

    sample_color = texelFetch(di, ivec2(sample_uv * gUniform.view_size), 0).xyz;
    vec3 pix_position = globalUniform.cameraData.cameraW
                + (-0.5 + sample_uv.x) * 2 * globalUniform.cameraData.cameraU
                + (-0.5 + sample_uv.y) * 2 * globalUniform.cameraData.cameraV;
    vec3 pix_dir = normalize(pix_position);
    float cos_0 = max(dot(-pix_dir, -normalize(globalUniform.cameraData.cameraW)), 0);
    float r_0 = length(pix_position);
    float area_0 = 4 * length(globalUniform.cameraData.cameraU) * length(globalUniform.cameraData.cameraV);
    float jacobian_0 = cos_0 * area_0 / (r_0 * r_0);

    float dist_1 = length(endState.info.samplePosInVS.xyz);
    vec3 dir_1 = normalize(endState.info.samplePosInVS.xyz);
    float cos_1 = max(dot(endState.normalInVS, -dir_1), 0);
    float jacobian_1 = jacobian_0 * (dist_1 * dist_1) / cos_1;
    
    if(dot(endState.normalInVS, -reflectionDirVS) < 0) return vec3(0);
    float cos_2 = max(dot(vec3(0,0,-1), -reflectionDirVS), 0);
    float dist_2 = max(length(endState.info.samplePosInVS - initState.info.samplePosInVS), 0);

    // vec3 sample_color = texture(di, sample_uv).xyz;
    if (dist_2 != 0 && !isinf(dist_2) && !isnan(dist_2) && !isinf(jacobian_1)) {
        float g = cos_2 / (dist_2 * dist_2);
        float jacobian_2 = jacobian_1 * g;
    
        float cos_3 = max(dot(reflectionDirVS, initState.normalInVS), 0);

        area_prob = jacobian_1;

        return sample_color * jacobian_2 * cos_3;
    }
    return vec3(0);
}

float g_term(
    in const vec2 sample_uv,
    in const InitStateTS initState,
    out float area_prob
) {
    area_prob = 0;
    
    InitStateTS endState;
    bool hasIsect = unpackVSInfo_Vec(sample_uv, endState);

    if (!hasIsect) {
        return (0);
    }

    // check visibility
    vec3 sample_color = vec3(0);
    vec3 reflectionDirVS = normalize(endState.info.samplePosInVS.xyz - initState.info.samplePosInVS.xyz);

    {
        SampleTS isect;
        uint result = FindOcclusion(initState, endState, isect);
        if(result == EnumIsectResult_None) {
         }
        else {
            return 0;
        }
    }
    
    if(dot(endState.normalInVS, -reflectionDirVS) < 0) return 0;
    float cos_2 = max(dot(vec3(0,0,-1), -reflectionDirVS), 0);
    float dist_2 = max(length(endState.info.samplePosInVS - initState.info.samplePosInVS), 0);
    return cos_2 / (dist_2 * dist_2);
}

vec2 sampleImportanceUV_v1(
    inout uint RNG,
    out float probability
) {
    // gImportanceMIP
    int mip_level = 11;
    vec2 uv = vec2(0.5, 0.5);
    ivec2 xy = ivec2(0);

    float p = 1.f;
    float pixel_size = 1. / 2;
    
    for (int i = 0; i < gUniform.is_depth; ++i) {
        pixel_size /= 2;    // half pixel size
        mip_level--; // next mip level
        xy *= 2; // sample next level

        // sample next level
        float rnd = UniformFloat(RNG);
        float accum = 0.f;
        int last_non_zero = -1;
        float last_non_zero_imp = 0;
        float last_non_zero_pdf = 0;
    
        vec4 child_importance;
        {   for (int j=0; j<4; ++j) {
                ivec2 xy_offset = four_neighbors_pix[j];
                float lumin_importance = texelFetch(importance_mip, xy + xy_offset, mip_level).x;
                child_importance[j] = lumin_importance;
            }
            const float sum = dot(vec4(1), child_importance);
            child_importance /= sum;
        }
        for (int j=0; j<4; ++j) {
            vec2 uv_offset = four_neighbors[j] * pixel_size;
            ivec2 xy_offset = four_neighbors_pix[j];
            float importance = child_importance[j];
            float nimportance = importance;
            accum += nimportance;
            
            if(nimportance > 0) {
                last_non_zero = j;
                last_non_zero_pdf = nimportance;
                last_non_zero_imp = importance;
            }

            if(rnd < accum) {
                uv = uv + uv_offset;
                xy = xy + xy_offset;
                p = p * nimportance;
                break;
            }
            else if(j==3 && last_non_zero!=-1) {
                uv = uv + four_neighbors[last_non_zero] * pixel_size;
                xy = xy + four_neighbors_pix[last_non_zero];
                p = p * last_non_zero_pdf;
                break;
            }
            else {
                // should not happen...
            }
        }
        // p *= 4;
    }


    // The depth value of the pixel:
    const float z = texelFetch(hi_z, xy, 0).r;
    const bool is_valid = (z < 1.0) ? true : false;
    // The position of the pixel in world space:
    const vec3 posInWS_0 = is_valid ? SS2WS(xy + vec2(0, 0), z) : vec3(0.0);
    const vec3 posInWS_1 = is_valid ? SS2WS(xy + vec2(0, 1), z) : vec3(0.0);
    const vec3 posInWS_2 = is_valid ? SS2WS(xy + vec2(1, 0), z) : vec3(0.0);
    const float area = length(cross(posInWS_1 - posInWS_0, posInWS_2 - posInWS_0));
    p /= area;
    
    vec2 uv_pertub = vec2(UniformFloat(RNG), UniformFloat(RNG)); // (0,1)
    uv_pertub = vec2(-1, -1) + uv_pertub * 2; // (-1, 1)
    uv += uv_pertub * pixel_size;
    probability = p;
    return uv;
}

SampleTS SampleTech_LightCut(
    in const InitStateTS state,
    inout uint RNG
) {
    SampleTS samplets;
    samplets.radiance = vec3(0);

    float sample_uv_pdf;
    // vec2 sample_uv = sampleImportanceUV_LC_v0_v1(RNG, sample_uv_pdf, state.info, state.normalInVS);

    // float sample_uv_pdf;
    {
        // vec2 uv = sampleImportanceUV_v1(RNG, sample_uv_pdf);
        vec2 uv = sampleImportanceUV_LC_v0_v1(RNG, sample_uv_pdf, state.info, state.normalInVS);
        
        ivec2 xy = ivec2(floor(uv * 2048));
        if(xy.x >= gUniform.view_size.x || xy.y >= gUniform.view_size.y) {
            samplets.radiance = vec3(1,0,1);
            return samplets;
        }
        {
            ivec2 fi_xy = xy;
            {   // The depth value of the pixel:
                const float z = texelFetch(hi_z, fi_xy, 0).r;
                const bool is_valid = (z < 1.0) ? true : false;
                // The position of the pixel in world space:
                const vec3 posInWS_0 = is_valid ? SS2WS(fi_xy + vec2(0, 0), z) : vec3(0.0);
                const vec3 posInWS_1 = is_valid ? SS2WS(fi_xy + vec2(0, 1), z) : vec3(0.0);
                const vec3 posInWS_2 = is_valid ? SS2WS(fi_xy + vec2(1, 0), z) : vec3(0.0);
                const float area = length(cross(posInWS_1 - posInWS_0, posInWS_2 - posInWS_0));
                sample_uv_pdf /= area;
            }
        }
        const float z = texelFetch(hi_z, xy, 0).r;
        samplets.xy = xy;
        samplets.uv = (xy + vec2(UniformFloat(RNG), UniformFloat(RNG))) / gUniform.view_size;
        samplets.hasIsect = unpackVSInfo_vec(uv, samplets.info);
        samplets.normalInVS = samplets.info.sampleNormalInVS;
    }

    const vec3 di_sample = texelFetch(di, samplets.xy, 0).rgb;
    if(di_sample == vec3(0)) return samplets;

    if(!samplets.hasIsect)
        return samplets;
    
    const vec3 dir = samplets.info.samplePosInVS.xyz - state.info.samplePosInVS.xyz;
    const float dist = length(dir);
    const vec3 dirInVS = normalize(dir);
    const float pdf = sample_uv_pdf;
    
    if(dot(dir, samplets.normalInVS) > 0.0) return samplets;

    // SampleTS isect;
    // uint result = FindIntersection(state, dir, isect);
    // if(result == EnumIsectResult_True) {
    //     vec2 offsetXY = abs(isect.uv - samplets.uv);
    //     offsetXY *= gUniform.view_size;
    //     InitStateTS realEndState;
    //     unpackVSInfo(isect.xy, realEndState);
    //     if(length(offsetXY) > 0.6) {
    //         return samplets;
    //     }
    // }
    // else return samplets;

    samplets.radiance = di_sample * max(dot(dirInVS, state.normalInVS), 0) *
            abs(dot(dirInVS, vec3(0,0,1))) / (pdf * dist * dist);

    return samplets;
}

float SampleTechPdf_LightCut(
    in const InitStateTS state,
    in const SampleTS samplets
) {
    const float uv_pdf = sampleImportanceUV_pdf(
        samplets.uv,
        state.info,
        state.normalInVS
    );
    float jacobian, sth;
    jacobian = jacobian_as2ss(state, samplets, sth);
    if(jacobian == 0) return 0;
    return uv_pdf / jacobian;
}

#endif