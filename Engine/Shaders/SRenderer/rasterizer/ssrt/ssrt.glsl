
vec2 saturate(vec2 v) { return clamp(v, 0.0, 1.0); }

// Returns the number of cells in the quad tree at the given level.
vec2 getCellCount(in const int mipLevel) {
    return textureSize(hi_z, mipLevel);
}

// Returns the 2D integer index of the cell that contains the given 2D position within it.
vec2 getCell(in const vec2 pos, in const vec2 cell_count) {
    return vec2(floor(pos * cell_count));
}

// Returns the 2D integer index of the cell that contains the given 2D position within it.
vec2 getCellCoord(in const vec2 pos, in const vec2 cell_count) {
    return vec2(pos * cell_count);
}

vec3 intersectDepthPlane(in const vec3 o, in const vec3 d, float z) {
	return o + d * z;
}

vec3 intersectCellBoundary(
    in const vec3 o, 
    in const vec3 d, 
    in const vec2 cell, 
    in const vec2 cell_count,
    in const vec2 crossStep,
    in const vec2 crossOffset
) { // crossStep is added to the current cell to get the next cell index
	vec2 index = cell + crossStep;
    // By dividing the cell index by cell count, we can get the position
    // of the boundaries between the current cell and the new cell .
	vec2 boundary = index / cell_count;
    // crossOffset is used to push the position just a tiny bit further
    // to make sure the new position is not right on the boundary.
	boundary += crossOffset;
	
    // delta between the new position and the origin
	vec2 delta = boundary - o.xy;
    // After the division, the x and y component in delta will have value between 0 to 1
    // which represents how far the delta position is from the origin of the ray.
	delta /= d.xy;
    // take the minimum of the two components,
    // x and y of delta because we want to cross the nearest boundary
	float t = min(delta.x, delta.y);
	
	return intersectDepthPlane(o, d, t);
}

bool crossedCellBoundary(
    in const vec2 cellIdxOne, 
    in const vec2 cellIdxTwo
) {
    return floor(cellIdxOne.x) != floor(cellIdxTwo.x) 
        || floor(cellIdxOne.y) != floor(cellIdxTwo.y);
}

float minimumDepthPlane(vec2 ray, int level, vec2 cell_count) {
    return texelFetch(hi_z, ivec2(ray.xy * cell_count), level).r;
}

vec3 SS2WS(in const vec2 xy, in const float z) {
    const vec4 posInCS =  vec4((xy/gUniform.view_size)*2-1.0f, z, 1) * vec4(1,1,1,1);
    vec4 posInVS = gUniform.InvProjMat * posInCS;
    posInVS /= posInVS.w;
    const vec4 posInWS = transpose(gUniform.TransInvViewMat) * vec4(posInVS.xyz, 1.0);
    return posInWS.xyz;
}

vec3 SS2VS(in const vec2 xy, in const float z) {
    const vec4 posInCS =  vec4((xy/gUniform.view_size)*2-1.0f, z, 1) * vec4(1,1,1,1);
    vec4 posInVS = gUniform.InvProjMat * posInCS;
    posInVS /= posInVS.w;
    return posInVS.xyz * vec3(1,-1,1);
}

struct SSRay {
    vec3 rayPosInTS;
    vec3 rayDirInTS;
    vec3 rayPosInVS;
    vec3 rayDirInVS;
    float maxDistance;
    float minDistance;
};

struct SSRTConfig {
    bool normal_check;
};

bool FindIntersection_HiZ(
    in const SSRay ss_ray,
    out vec3 intersection,
    inout SSRTConfig config
) {
    const vec3 samplePosInTS = ss_ray.rayPosInTS;
    const vec3 vReflDirInTS = ss_ray.rayDirInTS;
    const float maxTraceDistance = ss_ray.maxDistance;

    const int maxLevel = gUniform.hiz_mip_levels - 1;
	
    vec2 crossStep = vec2(vReflDirInTS.x>=0 ? 1 : -1, vReflDirInTS.y>=0 ? 1 : -1);
    vec2 crossOffset = crossStep / ivec2(gUniform.view_size) / 128;
    crossStep = saturate(crossStep);
    
    vec3 ray = samplePosInTS.xyz;
    float minZ = ray.z;
    float maxZ = ray.z + vReflDirInTS.z * maxTraceDistance;
    float minimumZ = ray.z + vReflDirInTS.z * ss_ray.minDistance;
    float deltaZ = (maxZ - minZ);

    const vec2 cellCount_0 = getCellCount(0);
    const ivec2 startXY = ivec2(getCell(ray.xy, cellCount_0));
    const vec3 startNormal = unpackNormal(texelFetch(ws_normal, startXY, 0).xyz);

    vec3 o = ray;
    vec3 d = vReflDirInTS * maxTraceDistance;
	
    int startLevel = 0;
    int stopLevel = 0;
    vec2 startCellCount = getCellCount(startLevel);
	
    // Move the current ray to the next cell in the reflection direction to avoid ‘self-intersection’.
    vec2 rayCell = getCell(ray.xy, startCellCount);
    ray = intersectCellBoundary(o, d, rayCell, startCellCount, crossStep, crossOffset*64);
    
    int level = startLevel;
    int iter = 0;
    bool isBackwardRay = vReflDirInTS.z<0;
    float rayDir = isBackwardRay ? -1 : 1;
    
    vec3 prevRay = ray;

    while(
        level >= stopLevel && 
        ray.z*rayDir <= maxZ*rayDir && 
        iter < gUniform.max_iteration
    ) {
        // get the cell number of the current ray

        const vec2 cellCount = getCellCount(level);
        const vec2 oldCellIdx = getCell(ray.xy, cellCount);
        
        // get the minimum depth of the current cell
        float cell_minZ = minimumDepthPlane(ray.xy, level, cellCount);
        // compare the min depth with the current depth of the ray. There can be two outcomes from the comparison.
        vec3 tmpRay = ((cell_minZ > ray.z) && !isBackwardRay) 
            // if the min depth is greater than the current depth, then we need to move the ray to the min depth.
            ? intersectDepthPlane(o, d, (cell_minZ - minZ)/deltaZ)
            // if the min depth is less than the current depth, then we keep the ray.
            : ray;
        
        const vec2 newCellIdx = getCell(tmpRay.xy, cellCount);
        
        float thickness = level == 0 ? (ray.z - cell_minZ) : 0;
        bool crossed = (isBackwardRay && (cell_minZ > ray.z)) 
                    || (thickness>(gUniform.max_thickness)) 
                    || crossedCellBoundary(oldCellIdx, newCellIdx);
        
        ivec2 xy = ivec2(getCell(tmpRay.xy, cellCount_0));
        float dist = length(vec2(xy.xy - startXY.xy));

        if(!crossed && level == 0) {
            bool larger_than_min = (ray.z*rayDir >= minimumZ*rayDir);

            bool si = false;

            if(!si) {
                if((cell_minZ > ray.z) && larger_than_min){
                    intersection = tmpRay;
                    if(config.normal_check) {
                        vec3 normalNow = unpackNormal(texelFetch(ws_normal, ivec2(intersection.xy * gUniform.view_size), 0).xyz);
                        if(dot(startNormal, normalNow) <= 0.999) {
                            return true;
                        }
                    }
                    else {
                        return true;
                    }
                }
                else {
                    intersection.xy = vec2(getCell(ray.xy, cellCount) + vec2(0.5))/cellCount;
                    if(config.normal_check) {
                        vec3 normalNow = unpackNormal(texelFetch(ws_normal, ivec2(intersection.xy * gUniform.view_size), 0).xyz);
                        if(dot(startNormal, normalNow) <= 0.999) {
                            return true;
                        }
                    }
                    else {
                        return true;
                    }
                }
            }
            // Move the current ray to the next cell in the reflection direction to avoid ‘self-intersection’.
            ray = intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep, crossOffset * 64);
        }
        else {
            prevRay = ray;
            ray = crossed
                ? intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep, crossOffset)
                : tmpRay;
            level = crossed 
                ? min(maxLevel, level+1) 
                : level-1;
        }

        ++iter;
    }

    return false;   // return intersected
}

/**
* Do DDA over the 2D plane.
* @ref: https://www.shadertoy.com/view/Nl23RV
*/
struct Trace2D_Linear {
    vec2 ro;    // ray origin
    vec2 rd;    // ray direction, normalized
    vec2 dd;    // 1/abs(rd)
    ivec2 sd;   // sign(rd)
	float t0;   // range start
	float t1;   // range end
};

struct DDA2D_Linear {
	vec2 st;    // t of next axis boundary
	ivec2 mp;   // current cell index
	float t;    // current trace at
};

struct DebugPack {
    ivec2 qi;
    vec2 pixel;
    float cd;
    bool celltouchesline;
};

void setup(
    vec2 pa, vec2 pb,
    out DDA2D_Linear dda,
    out Trace2D_Linear tr
) {
    // Trace2D_Linear setup
	tr.ro = pa;
	tr.rd = pb - pa;
	vec2 ad = abs(tr.rd);
    // sign() is somewhat buggy, can sometimes produce values *slightly* below 1.0 in magnitude, which convert to integer zero... many workarounds
	tr.sd = ivec2(sign(tr.rd) * 1.001);
	tr.t1 = length(tr.rd); // could use ad but abs isn't necessary here
	tr.t0 = 0.;
	tr.rd /= tr.t1; //tr.rd = normalize(tr.rd);
	ad /= tr.t1;
	tr.dd = vec2(1) / max(ad, 1e-32); // will be effectively infinity where rd was 0, so only use when actually crossing between cells
	// DDA2D_Linear setup
    dda.t = 0.;
	dda.mp = ivec2(floor(tr.ro)); // current cell index
	dda.st = tr.ro - vec2(dda.mp); //fract(tr.ro);
	dda.st = (.5 - dda.st) * vec2(tr.sd) + .5;
	// compensate when starting precisely on an integer (grid cell boundary) going in negative direction, would be otherwise off by one
	ivec2 sot = ivec2(equal(dda.st, vec2(0))); // mask for on cell boundary (integer)
	dda.mp += sot * min(tr.sd, ivec2(0)); // factor mask for negative directions
	dda.st += vec2(sot); // step cannot be zero.  If start on boundary, just go to next one!
	dda.st *= tr.dd;
}

bool traverse(
    inout DDA2D_Linear dda,
    inout Trace2D_Linear tr
    #ifdef DEBUG
    , inout DebugPack debug
    #endif
) {
    #ifdef DEBUG
    bool r = dda.t < tr.t1;
	if (r) {
        if (dda.t > 0.) // no dot at ray origin please
            debug.cd = min(debug.cd, distance(debug.pixel, tr.ro + tr.rd * dda.t));
        if (debug.qi == dda.mp)
            debug.celltouchesline = true; // color cell later
	}
	return r;
    #endif
	return dda.t < tr.t1;
}

int nextIsect(
    inout DDA2D_Linear dda, 
    in const Trace2D_Linear tr)
{
	int s = dda.st.x <= dda.st.y ? 0 : 1; // which cell boundary is closer?
	dda.t = dda.st[s]; // update total distance traversed
	dda.st[s] += tr.dd[s]; // step to next boundary 
	dda.mp[s] += tr.sd[s]; // move, which way? to next cell
	return s;
}

vec2 Scan(
    inout DDA2D_Linear dda, 
    Trace2D_Linear tr, 
    out ivec2 n
    #ifdef DEBUG
    , inout DebugPack debug
    #endif
) { 
    float ot = dda.t;
	int s;
	while (traverse(dda, tr
    #ifdef DEBUG
    , debug
    #endif
    ))
		s = nextIsect(dda, tr);
	n = ivec2(0);
	n[s] = -tr.sd[s]; // avoid negating zeroes
	return vec2(ot, dda.t);
}

float ScanDDA2(in const vec2 a, in const vec2 b
    #ifdef DEBUG
    , inout DebugPack debug
    #endif
) {
    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(a, b, dda, tr);
    ivec2 n;
    return Scan(dda, tr, n
        #ifdef DEBUG
        , debug
        #endif
    ).y;
}

bool FindIntersection_Linear(
    in const SSRay ss_ray,
    out vec3 intersection
) {
    const vec3 samplePosInTS = ss_ray.rayPosInTS;
    const vec3 vReflDirInTS = ss_ray.rayDirInTS;
    const float maxTraceDistance = ss_ray.maxDistance;

    const vec3 vReflectionEndPosInTS = samplePosInTS + vReflDirInTS * maxTraceDistance;

    const vec2 cellCount = getCellCount(0);

    const vec2 m0 = getCellCoord(samplePosInTS.xy, cellCount);
    const vec2 m1 = getCellCoord(vReflectionEndPosInTS.xy, cellCount);

    // intersection = vec3(samplePosInTS.xy, 0);
    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(m0, m1, dda, tr);
    ivec2 n;

    return false;
}

struct RayStartInfo {
    vec4 samplePosInCS;
    vec4 samplePosInVS;
    vec3 samplePosInTS;
    vec3 samplePosInWS;
    vec3 sampleNormalInVS;
    vec3 sampleNormalInWS;
};

bool unpackVSInfo(
    in const uvec2 tid,
    out RayStartInfo rayStartInfo
) {
    // Unpack the depth from the texture.
    float sampleDepth = texelFetch(hi_z, ivec2(tid), 0).x;
    if(sampleDepth == 1) return false;

    // Unpack the normal from the texture.
    rayStartInfo.sampleNormalInWS = unpackNormal(texelFetch(ws_normal, ivec2(tid), 0).xyz);
    rayStartInfo.sampleNormalInVS = normalize((gUniform.TransInvViewMat * vec4(rayStartInfo.sampleNormalInWS, 0)).xyz);
    rayStartInfo.sampleNormalInVS.y = -rayStartInfo.sampleNormalInVS.y;
    
    // From the depth, compute the position in clip space.
    rayStartInfo.samplePosInCS =  vec4(((vec2(tid)+0.5)/gUniform.view_size)*2-1.0f, sampleDepth, 1);
    rayStartInfo.samplePosInCS.y *= -1;
    // From the depth, compute the position in view space.
    rayStartInfo.samplePosInVS = gUniform.InvProjMat * rayStartInfo.samplePosInCS;
    rayStartInfo.samplePosInVS /= rayStartInfo.samplePosInVS.w;

    // rayStartInfo.samplePosInVS.xyz += sampleNormalInVS * 0.01f;
    // Texture space
    rayStartInfo.samplePosInTS = rayStartInfo.samplePosInCS.xyz;
    rayStartInfo.samplePosInTS.xy *= vec2(0.5f, -0.5f);
    rayStartInfo.samplePosInTS.xy += vec2(0.5f, 0.5f);
    // World space
    rayStartInfo.samplePosInWS = SS2WS((tid + 0.5), sampleDepth);
    return true;
}

bool unpackVSInfo_vec(
    in const vec2 uv,
    out RayStartInfo rayStartInfo
) {
    // Unpack the depth from the texture.
    ivec2 tid = ivec2(floor(uv * gUniform.view_size));
    float sampleDepth = texelFetch(hi_z, ivec2(tid), 0).x;
    if(sampleDepth == 1) return false;

    // Unpack the normal from the texture.
    rayStartInfo.sampleNormalInWS = unpackNormal(texelFetch(ws_normal, ivec2(tid), 0).xyz);
    rayStartInfo.sampleNormalInVS = normalize((gUniform.TransInvViewMat * vec4(rayStartInfo.sampleNormalInWS, 0)).xyz);
    rayStartInfo.sampleNormalInVS.y = -rayStartInfo.sampleNormalInVS.y;
    
    // From the depth, compute the position in clip space.
    rayStartInfo.samplePosInCS =  vec4(uv*2-1.0f, sampleDepth, 1);
    rayStartInfo.samplePosInCS.y *= -1;
    // From the depth, compute the position in view space.
    rayStartInfo.samplePosInVS = gUniform.InvProjMat * rayStartInfo.samplePosInCS;
    rayStartInfo.samplePosInVS /= rayStartInfo.samplePosInVS.w;

    // rayStartInfo.samplePosInVS.xyz += sampleNormalInVS * 0.01f;
    // Texture space
    rayStartInfo.samplePosInTS = rayStartInfo.samplePosInCS.xyz;
    rayStartInfo.samplePosInTS.xy *= vec2(0.5f, -0.5f);
    rayStartInfo.samplePosInTS.xy += vec2(0.5f, 0.5f);
    // World space
    rayStartInfo.samplePosInWS = SS2WS((tid + 0.5), sampleDepth);
    return true;
}

// Compute the position, the reflection direction, maxTraceDistance of the sample in texture space.
bool ComputePosAndReflection(
    in  const uvec2 tid,
    in  const vec3  vSampleNormalInVS,
    out vec3  outSamplePosInTS,
    out vec3  outReflDirInTS,
    out float outMaxDistance
) {
    float sampleDepth = texelFetch(hi_z, ivec2(tid), 0).x;
    if(sampleDepth == 1) return false;

    vec4 samplePosInCS =  vec4(((vec2(tid)+0.5)/gUniform.view_size)*2-1.0f, sampleDepth, 1);
    samplePosInCS.y *= -1;

    vec4 samplePosInVS = gUniform.InvProjMat * samplePosInCS;
    samplePosInVS /= samplePosInVS.w;

    vec3 vCamToSampleInVS = normalize(samplePosInVS.xyz);
    vec4 vReflectionInVS = vec4(reflect(vCamToSampleInVS.xyz, vSampleNormalInVS.xyz),0);

    vec4 vReflectionEndPosInVS = samplePosInVS + vReflectionInVS * 1000;
    vReflectionEndPosInVS /= (vReflectionEndPosInVS.z < 0 ? vReflectionEndPosInVS.z : 1);
    vec4 vReflectionEndPosInCS = gUniform.ProjMat * vec4(vReflectionEndPosInVS.xyz, 1);
    vReflectionEndPosInCS /= vReflectionEndPosInCS.w;
    vec3 vReflectionDir = normalize((vReflectionEndPosInCS - samplePosInCS).xyz);

    // Transform to texture space
    samplePosInCS.xy *= vec2(0.5f, -0.5f);
    samplePosInCS.xy += vec2(0.5f, 0.5f);
    
    vReflectionDir.xy *= vec2(0.5f, -0.5f);
    
    outSamplePosInTS = samplePosInCS.xyz;
    outReflDirInTS = vReflectionDir;
    
	// Compute the maximum distance to trace before the ray goes outside of the visible area.
    outMaxDistance = outReflDirInTS.x>=0 ? (1 - outSamplePosInTS.x)/outReflDirInTS.x  : -outSamplePosInTS.x/outReflDirInTS.x;
    outMaxDistance = min(outMaxDistance, outReflDirInTS.y<0 ? (-outSamplePosInTS.y/outReflDirInTS.y)  : ((1-outSamplePosInTS.y)/outReflDirInTS.y));
    outMaxDistance = min(outMaxDistance, outReflDirInTS.z<0 ? (-outSamplePosInTS.z/outReflDirInTS.z) : ((1-outSamplePosInTS.z)/outReflDirInTS.z));

    return true;
}

SSRay PrepareSSRT(
    in  const RayStartInfo startInfo,
    in  const vec3  reflDirInVS
) {
    const vec4 vReflectionInVS = vec4(reflDirInVS,0);
    // Compute the reflection direction in clip space.
    const float signRefl = vReflectionInVS.z < 0 ? -1 : 1;
    vec4 vReflectionEndPosInVS = startInfo.samplePosInVS + vReflectionInVS * 10 * signRefl;
    vec4 vReflectionEndPosInWS = startInfo.samplePosInVS + vReflectionInVS * 10;
    // transpose(gUniform.TransInvViewMat) * vec4(posInVS.xyz, 1.0);
    // vReflectionEndPosInVS /= (vReflectionEndPosInVS.z < 0 ? vReflectionEndPosInVS.z : 1);
    vec4 vReflectionEndPosInCS = gUniform.ProjMat * vec4(vReflectionEndPosInVS.xyz, 1);
    vReflectionEndPosInCS /= vReflectionEndPosInCS.w;
    vec3 vReflectionDir = normalize((vReflectionEndPosInCS - startInfo.samplePosInCS).xyz * signRefl);
    // Transform to texture space    
    vReflectionDir.xy *= vec2(0.5f, -0.5f);
    vec3 outReflDirInTS = vReflectionDir;
	// Compute the maximum distance to trace before the ray goes outside of the visible area.
    float outMaxDistance = outReflDirInTS.x>=0 ? (1 - startInfo.samplePosInTS.x)/outReflDirInTS.x  : -startInfo.samplePosInTS.x/outReflDirInTS.x;
    outMaxDistance = min(outMaxDistance, outReflDirInTS.y<0 ? (-startInfo.samplePosInTS.y/outReflDirInTS.y)  : ((1-startInfo.samplePosInTS.y)/outReflDirInTS.y));
    outMaxDistance = min(outMaxDistance, outReflDirInTS.z<0 ? (-startInfo.samplePosInTS.z/outReflDirInTS.z) : ((1-startInfo.samplePosInTS.z)/outReflDirInTS.z));
    SSRay ray;
    ray.rayPosInVS =  startInfo.samplePosInVS.xyz;
    ray.rayDirInVS = normalize(vReflectionEndPosInWS.xyz - startInfo.samplePosInVS.xyz);
    ray.rayPosInTS = startInfo.samplePosInTS;
    ray.rayDirInTS = outReflDirInTS;
    ray.maxDistance = outMaxDistance;
    ray.minDistance = 0;
    return ray;
}

SSRay PrepareSSRT(
    in  const RayStartInfo startInfo,
    in  const RayStartInfo endInfo
) {
    vec3 vReflectionDir = (endInfo.samplePosInCS - startInfo.samplePosInCS).xyz;
    const float dirLength = length(vReflectionDir);
    vReflectionDir /= dirLength;
    // Transform to texture space    
    vReflectionDir.xy *= vec2(0.5f, -0.5f);
    vec3 outReflDirInTS = vReflectionDir;
	// Compute the maximum distance to trace before the ray goes outside of the visible area.
    float outMaxDistance = outReflDirInTS.x>=0 ? (1 - startInfo.samplePosInTS.x)/outReflDirInTS.x  : -startInfo.samplePosInTS.x/outReflDirInTS.x;
    outMaxDistance = min(outMaxDistance, outReflDirInTS.y<0 ? (-startInfo.samplePosInTS.y/outReflDirInTS.y)  : ((1-startInfo.samplePosInTS.y)/outReflDirInTS.y));
    outMaxDistance = min(outMaxDistance, outReflDirInTS.z<0 ? (-startInfo.samplePosInTS.z/outReflDirInTS.z) : ((1-startInfo.samplePosInTS.z)/outReflDirInTS.z));
    outMaxDistance = min(dirLength, outMaxDistance);
    SSRay ray;
    ray.rayPosInTS = startInfo.samplePosInTS;
    ray.rayDirInTS = outReflDirInTS;
    ray.maxDistance = outMaxDistance;
    ray.minDistance = 0;
    return ray;
}



/*****************************************************
**               Find Ray Intersection              **
*****************************************************/

bool FindIntersection_DDA(
    in const SSRay ray,
    out vec3 intersection,
    inout SSRTConfig config
) {
    const vec2 cellCount = getCellCount(0);

    const vec3 vReflectionEndPosInTS = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
    const vec2 m0 = getCellCoord(ray.rayPosInTS.xy, cellCount);
    const vec2 m1 = getCellCoord(vReflectionEndPosInTS.xy, cellCount);

    // DDA setup
    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(m0, m1, dda, tr);
    nextIsect(dda, tr);
    
    const vec3 dir_norm = ray.rayDirInTS * abs(ray.maxDistance / tr.t1);
    
    int iter = 0;
    float z_cell = texelFetch(hi_z, dda.mp, 0).r;
    float z_cell_prev = z_cell;
    vec3 point = ray.rayPosInTS + dda.t * dir_norm;

    while (traverse(dda, tr)) {
        if(iter > gUniform.max_iteration) break;
        
        // take one step forward
        nextIsect(dda, tr);

        if(dda.mp.x < 0 || dda.mp.x >= cellCount.x || dda.mp.y < 0 || dda.mp.y >= cellCount.y) {
            return false;
        }

        const vec3 point_new = ray.rayPosInTS + dda.t * dir_norm;
        const float z_old = point.z;
        const float z_new = point_new.z;
        
        if((z_cell - z_old) * (z_cell - z_new) < 0) {
            // Find intersection
            intersection = 0.5 * (point_new + point);
            return true;
        }

        // update cell_z
        z_cell_prev = z_cell;
        z_cell = texelFetch(hi_z, dda.mp, 0).r;
        point = point_new;
        iter++;
    }
    
    return false;
}

bool FindIntersection_DDA_Connected(
    in const SSRay ray,
    out vec3 intersection,
    inout SSRTConfig config
) {
    const vec2 cellCount = getCellCount(0);

    const vec3 vReflectionEndPosInTS = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
    const vec2 m0 = getCellCoord(ray.rayPosInTS.xy, cellCount);
    const vec2 m1 = getCellCoord(vReflectionEndPosInTS.xy, cellCount);

    // DDA setup
    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(m0, m1, dda, tr);
    nextIsect(dda, tr);
    
    const vec3 dir_norm = ray.rayDirInTS * abs(ray.maxDistance / tr.t1);
    
    int iter = 0;
    float z_cell = texelFetch(hi_z, dda.mp, 0).r;
    float z_cell_prev = z_cell;
    vec3 point = ray.rayPosInTS + dda.t * dir_norm;

    while (traverse(dda, tr)) {
        if(iter > gUniform.max_iteration) break;
        
        // take one step forward
        nextIsect(dda, tr);

        if(dda.mp.x < 0 || dda.mp.x >= cellCount.x || dda.mp.y < 0 || dda.mp.y >= cellCount.y) {
            return false;
        }

        const vec3 point_new = ray.rayPosInTS + dda.t * dir_norm;
        const float z_old = point.z;
        const float z_new = point_new.z;
        
        if((z_cell - z_old) * (z_cell - z_new) < 0) {
            // Find intersection
            intersection = 0.5 * (point_new + point);
            return true;
        }
        if(z_old >= z_cell && z_new >= z_cell) {
            // Find intersection
            intersection = 0.5 * (point_new + point);
            return true;
        }

        // update cell_z
        z_cell_prev = z_cell;
        z_cell = texelFetch(hi_z, dda.mp, 0).r;
        point = point_new;
        iter++;
    }
    
    return false;
}

bool FindOcclusion_DDA(
    in const SSRay ray,
    in const vec2 end_uv,
    out vec3 intersection,
    inout SSRTConfig config
) {
    const vec2 cellCount = getCellCount(0);

    const vec3 vReflectionEndPosInTS = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
    const vec2 m0 = getCellCoord(ray.rayPosInTS.xy, cellCount);
    const vec2 m1 = getCellCoord(end_uv, cellCount);

    // DDA setup
    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(m0, m1, dda, tr);
    nextIsect(dda, tr);
    
    const vec3 dir_norm = ray.rayDirInTS * abs(ray.maxDistance / tr.t1);
    
    int iter = 0;
    float z_cell = texelFetch(hi_z, dda.mp, 0).r;
    float z_cell_prev = z_cell;
    vec3 point = ray.rayPosInTS + dda.t * dir_norm;

    while (traverse(dda, tr)) {
        if(iter > gUniform.max_iteration) break;
        
        // take one step forward
        nextIsect(dda, tr);

        if(dda.mp.x < 0 || dda.mp.x >= cellCount.x || dda.mp.y < 0 || dda.mp.y >= cellCount.y) {
            return false;
        }

        const vec3 point_new = ray.rayPosInTS + dda.t * dir_norm;
        const float z_old = point.z;
        const float z_new = point_new.z;
        
        if(iter > 1) {
            if((z_cell - z_old) * (z_cell - z_new) < 0) {
                // Find intersection
                const float a = abs(z_cell - z_old);
                const float b = abs(z_cell - z_new);
                intersection = (point * b + point_new * a) / (a + b);
                return true;
            }
        }
        // update cell_z
        z_cell_prev = z_cell;
        z_cell = texelFetch(hi_z, dda.mp, 0).r;
        point = point_new;
        iter++;
    }
    
    return false;
}

vec3[4] FetchRectPointsSS(ivec2 center) {
    vec2 toUV = vec2(1./gUniform.view_size);
    vec4 zQuad = textureGather(hi_z, vec2(center * toUV));
    vec3 points[4];
    points[0] = vec3(center.xy + vec2(-0.5, -0.5), zQuad.w) * vec3(toUV, 1);
    points[1] = vec3(center.xy + vec2(-0.5, +0.5), zQuad.x) * vec3(toUV, 1);
    points[2] = vec3(center.xy + vec2(+0.5, -0.5), zQuad.z) * vec3(toUV, 1);
    points[3] = vec3(center.xy + vec2(+0.5, +0.5), zQuad.y) * vec3(toUV, 1);
    return points;
}

bool FindIntersection_DDA_Trianglulate_Safe(
    in const SSRay ray,
    out vec3 intersection,
    inout SSRTConfig config
) {
    const vec3 roWS = ray.rayPosInVS;
    const vec3 rdWS = ray.rayDirInVS;

    const vec2 cellCount = getCellCount(0);

    const vec3 vReflectionEndPosInTS = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
    const vec2 m0 = getCellCoord(ray.rayPosInTS.xy, cellCount) - vec2(0.5);
    const vec2 m1 = getCellCoord(vReflectionEndPosInTS.xy, cellCount) - vec2(0.5);

    // DDA setup
    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(m0, m1, dda, tr);
    nextIsect(dda, tr);
    
    // const vec3 dir_norm = ray.rayDirInTS * abs(ray.maxDistance / tr.t1);
    
    // float z_cell = texelFetch(hi_z, dda.mp, 0).r;
    // float z_cell_prev = z_cell;
    // vec3 point = ray.rayPosInTS + dda.t * dir_norm;
    ivec2 center = dda.mp;
    nextIsect(dda, tr);
    int iter = 2;

    while (traverse(dda, tr)) {
        if(iter > gUniform.max_iteration) break;
        
        if(center.x < 1 || center.x >= cellCount.x || center.y < 1 || center.y >= cellCount.y) {
            return false;
        }

        // The depth value of the pixel:
        const float z = texelFetch(hi_z, center, 0).r;
        const bool is_valid = (z < 1.0) ? true : false;
        // The position of the pixel in world space:
        const vec3 posInWS_0 = is_valid ? SS2VS(center + vec2(0.5, 0.5), z) : vec3(0.0);
        // Fetch right pixel
        const bool is_valid_right = (center.x + 1 < cellCount.x) ? true : false;
        const float z_right = is_valid_right ? texelFetch(hi_z, center + ivec2(1, 0), 0).r : 0.0;
        const vec3 posInWS_1 = is_valid_right ? SS2VS(center + vec2(1.5, 0.5), z_right) : posInWS_0;
        // Fetch bottom pixel
        const bool is_valid_down = (center.y + 1 < cellCount.y) ? true : false;
        const float z_down = is_valid_down ? texelFetch(hi_z, center + ivec2(0, 1), 0).r : 0.0;
        const vec3 posInWS_2 = is_valid_down ? SS2VS(center + vec2(0.5, 1.5), z_down) : posInWS_0;
        // Fetch right-bottom pixel
        const bool is_valid_rd = (center.x + 1 < cellCount.x && center.y + 1 < cellCount.y) ? true : false;
        const float z_rd = is_valid_rd ? texelFetch(hi_z, center + ivec2(1, 1), 0).r : 0.0;
        const vec3 posInWS_3 = is_valid_rd ? SS2VS(center + vec2(1.5, 1.5), z_rd) : posInWS_0;
        
        vec3[4] points;
        points[0] = posInWS_0;
        points[1] = posInWS_1;
        points[2] = posInWS_2;
        points[3] = posInWS_3;

        vec3 barycentricCoord;
        bool hit = ray_triangle_intersection_pbrt(
            roWS, rdWS, 0, k_inf, 
            points[0], points[1], points[2], barycentricCoord);
        if(hit) {
            intersection = points[0] * barycentricCoord.x + 
                points[1] * barycentricCoord.y +
                points[2] * barycentricCoord.z;
            intersection = vec3(vec2(center)/gUniform.view_size, 1);
            return true; 
        }
        hit = ray_triangle_intersection_pbrt(
            roWS, rdWS, 0, k_inf, 
            points[2], points[1], points[3], barycentricCoord);
        if(hit) {
            intersection = points[0] * barycentricCoord.x + 
                points[1] * barycentricCoord.y +
                points[2] * barycentricCoord.z;
            intersection = vec3(vec2(center)/gUniform.view_size, 1);
            return true; 
        }

        center = dda.mp;
        // take one step forward
        nextIsect(dda, tr);

        iter++;
    }
    
    return false;
}

bool FindIntersection_DDA_Trianglulate(
    in const SSRay ray,
    out vec3 intersection,
    inout SSRTConfig config
) {
    const vec2 cellCount = getCellCount(0);

    const vec3 vReflectionEndPosInTS = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
    const vec2 m0 = getCellCoord(ray.rayPosInTS.xy, cellCount) - vec2(0.5);
    const vec2 m1 = getCellCoord(vReflectionEndPosInTS.xy, cellCount) - vec2(0.5);

    // DDA setup
    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(m0, m1, dda, tr);
    nextIsect(dda, tr);
    
    // const vec3 dir_norm = ray.rayDirInTS * abs(ray.maxDistance / tr.t1);
    
    // float z_cell = texelFetch(hi_z, dda.mp, 0).r;
    // float z_cell_prev = z_cell;
    // vec3 point = ray.rayPosInTS + dda.t * dir_norm;
    ivec2 center = dda.mp + ivec2(1);
    nextIsect(dda, tr);
    int iter = 2;

    while (traverse(dda, tr)) {
        if(iter > gUniform.max_iteration) break;
        
        if(center.x < 1 || center.x >= cellCount.x || center.y < 1 || center.y >= cellCount.y) {
            return false;
        }

        vec3[4] points = FetchRectPointsSS(center);

        points[0] = vec3(0,0,1);
        points[1] = vec3(1,0,1);
        points[2] = vec3(0,1,1);
        points[3] = vec3(1,1,1);

        vec3 barycentricCoord;
        bool hit = ray_triangle_intersection_pbrt(
            ray.rayPosInTS, normalize(ray.rayDirInTS), 0, k_inf, 
            points[0], points[1], points[2], barycentricCoord);
        if(hit) {
            intersection = points[0] + 
                (points[1] - points[0]) * barycentricCoord.x +
                (points[2] - points[0]) * barycentricCoord.y;
            intersection = vec3(vec2(center)/gUniform.view_size, 1);
            return true; 
        }
        hit = ray_triangle_intersection_pbrt(
                    ray.rayPosInTS, normalize(ray.rayDirInTS), 0, k_inf, 
                    points[2], points[1], points[3], barycentricCoord);
        if(hit) {
            intersection = points[0] + 
                (points[1] - points[2]) * barycentricCoord.x +
                (points[3] - points[2]) * barycentricCoord.y;
            intersection = vec3(vec2(center)/gUniform.view_size, 1);
            return true; 
        }


        center = dda.mp + ivec2(1);
        // take one step forward
        nextIsect(dda, tr);

        iter++;
    }
    
    return false;
}