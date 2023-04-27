
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

struct SSRay {
    vec3 rayPosInTS;
    vec3 rayDirInTS;
    float maxDistance;
    float minDistance;
};


bool FindIntersection_HiZ(
    in const SSRay ss_ray,
    out vec3 intersection
) {
    const vec3 samplePosInTS = ss_ray.rayPosInTS;
    const vec3 vReflDirInTS = ss_ray.rayDirInTS;
    const float maxTraceDistance = ss_ray.maxDistance;


    const int maxLevel = pushConstants.hiz_mip_levels - 1;
	
    vec2 crossStep = vec2(vReflDirInTS.x>=0 ? 1 : -1, vReflDirInTS.y>=0 ? 1 : -1);
    vec2 crossOffset = crossStep / ivec2(pushConstants.view_size) / 128;
    crossStep = saturate(crossStep);
    
    vec3 ray = samplePosInTS.xyz;
    float minZ = ray.z;
    float maxZ = ray.z + vReflDirInTS.z * maxTraceDistance;
    float minimumZ = ray.z + vReflDirInTS.z * ss_ray.minDistance;
    float deltaZ = (maxZ - minZ);

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
        iter < pushConstants.max_iteration
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
                    || (thickness>(pushConstants.max_thickness)) 
                    || crossedCellBoundary(oldCellIdx, newCellIdx);
   
        if(!crossed && level == 0) {
            bool larger_than_min = (ray.z*rayDir >= minimumZ*rayDir);
            if((cell_minZ > ray.z) && larger_than_min){
                intersection = tmpRay;
                return true;
            }
            else if (larger_than_min) {
                intersection.xy = vec2(getCell(ray.xy, cellCount) + vec2(0.5))/cellCount;
                return true;
            }
        }
        
        prevRay = ray;
        ray = crossed
            ? intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep, crossOffset)
            : tmpRay;
        level = crossed 
            ? min(maxLevel, level+1) 
            : level-1;

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

    // float ray_z_debug = 0;
    // // float l = ScanDDA2(m0, m1, debugPack);
    // float l = 0.f;
    // DDA2D_Linear dda;
    // Trace2D_Linear tr;
    // setup(m0, m1, dda, tr);
    // ivec2 n;
    // vec3 pos_prev = samplePosInTS;
    // vec2 uv_prev = tr.ro;
    // const vec3 dir_norm = vReflDirInTS * maxTraceDistance / tr.t1;

    // float t_prev = 0.f;
    // ivec2 prev_mp = dda.mp;
    // {
    //     float ot = dda.t;
    //     int s;
    //     while (true) {
    //         bool go = traverse(dda, tr, debugPack);

    //         vec3 pos = samplePosInTS + (dda.t + t_prev) * .5 * dir_norm;
    //         // vec2 uv = pos.xy * .5 + pos_prev.xy * .5;
    //         vec2 uv = tr.ro + tr.rd * (dda.t + t_prev) * .5;
    //         // float z = minimumDepthPlane(pos.xy, level, cellCount);
    //         float z = pos.z;
    //         // float z = minimumDepthPlane((uv) / cellCount, 4, cellCount);
    //         // float z = texelFetch(hi_z, dda.mp, 4).r;
    //         // pos_prev = pos;
    //         t_prev = dda.t;
    //         uv_prev = uv;

    //         if(debugPack.qi == prev_mp) {
    //             ray_z_debug = z;
    //         }
    //         prev_mp = dda.mp;
    //         if(!go) break;
    //         s = nextIsect(dda, tr);
    //     }
    //     n = ivec2(0);
    //     n[s] = -tr.sd[s]; // avoid negating zeroes
    //     l = vec2(ot, dda.t).y;
    // }
    
    return false;
	// return vec2(ot, dda.t);


    // // dp is the vector that moves the current ray to its next position by adding it to the current position of the ray.
    // vec3 dp = vReflectionEndPosInTS.xyz - samplePosInTS.xyz;
    // ivec2 sampleScreenPos = ivec2(samplePosInTS.xy * pushConstants.view_size.xy);
    // ivec2 endPosScreenPos = ivec2(vReflectionEndPosInTS.xy * pushConstants.view_size.xy);
    // ivec2 dp2 = endPosScreenPos - sampleScreenPos;
    // const int max_dist = max(abs(dp2.x), abs(dp2.y));
    // dp /= max_dist;
    
    // vec4 rayPosInTS = vec4(samplePosInTS.xyz + dp, 0);
    // vec4 vRayDirInTS = vec4(dp.xyz, 0);
	// vec4 rayStartPos = rayPosInTS;

    // int hitIndex = -1;
    // for(int i = 0;i<=max_dist && i<pushConstants.max_iteration; i += 4)  {
    //     float depth0 = 0;
    //     float depth1 = 0;
    //     float depth2 = 0;
    //     float depth3 = 0;

    //     vec4 rayPosInTS0 = rayPosInTS+vRayDirInTS*0;
    //     vec4 rayPosInTS1 = rayPosInTS+vRayDirInTS*1;
    //     vec4 rayPosInTS2 = rayPosInTS+vRayDirInTS*2;
    //     vec4 rayPosInTS3 = rayPosInTS+vRayDirInTS*3;

    //     depth3 = texture(hi_z, rayPosInTS3.xy).x;
    //     depth2 = texture(hi_z, rayPosInTS2.xy).x;
    //     depth1 = texture(hi_z, rayPosInTS1.xy).x;
    //     depth0 = texture(hi_z, rayPosInTS0.xy).x;

    //     {   float thickness = rayPosInTS3.z - depth3;
    //         hitIndex = (thickness>=0 
    //         && thickness < pushConstants.max_thickness
    //         && abs(samplePosInTS.z - depth3)>pushConstants.min_thickness) ? (i+3) : hitIndex;
    //     } {
    //         float thickness = rayPosInTS2.z - depth2;
    //         hitIndex = (thickness>=0 
    //         && thickness < pushConstants.max_thickness
    //         && abs(samplePosInTS.z - depth2)>pushConstants.min_thickness
    //         ) ? (i+2) : hitIndex;
    //         uv = vec4(rayPosInTS2.xyz, depth2);
    //     } {
    //         float thickness = rayPosInTS1.z - depth1;
    //         hitIndex = (thickness>=0 
    //         && thickness < pushConstants.max_thickness
    //         && abs(samplePosInTS.z - depth1)>pushConstants.min_thickness
    //         ) ? (i+1) : hitIndex;
    //         uv = vec4(rayPosInTS1.xyz, depth1);
    //     } {
    //         float thickness = rayPosInTS0.z - depth0;
    //         hitIndex = (thickness>=0 
    //         && thickness < pushConstants.max_thickness
    //         && abs(samplePosInTS.z - depth0)>pushConstants.min_thickness
    //         ) ? (i+0) : hitIndex;
    //         uv = vec4(rayPosInTS0.xyz, depth0);
    //     }

    //     if(hitIndex != -1) break;

    //     rayPosInTS = rayPosInTS3 + vRayDirInTS;
    // }

    // bool intersected = hitIndex >= 0;
    // intersection = rayStartPos.xyz + vRayDirInTS.xyz * hitIndex;

    // const vec2 cellCount = getCellCount(0); 
    // const vec2 oldCellIdx = getCell(intersection.xy, cellCount);    
    // iter = hitIndex;
    // return intersected;
}

// vec3 packNormal(in const vec3 normal) {
//     return 0.5 * vec3(normal + 1.0);
// }

// vec3 unpackNormal(in const vec3 normal) {
//     return normalize(normal * 2.0 - 1.0);
// }

struct RayStartInfo {
    vec4 samplePosInCS;
    vec4 samplePosInVS;
    vec3 samplePosInTS;
};

bool unpackVSInfo(
    in const uvec2 tid,
    out vec3 sampleNormalInVS,
    out RayStartInfo rayStartInfo
) {
    // Unpack the depth from the texture.
    float sampleDepth = texelFetch(hi_z, ivec2(tid), 0).x;
    if(sampleDepth == 1) return false;

    // Unpack the normal from the texture.
    vec3 sampleNormalInWS = unpackNormal(texelFetch(ws_normal, ivec2(tid), 0).xyz);
    sampleNormalInVS = normalize((pushConstants.TransInvViewMat * vec4(sampleNormalInWS, 0)).xyz);
    sampleNormalInVS.y = -sampleNormalInVS.y;
    
    // From the depth, compute the position in clip space.
    rayStartInfo.samplePosInCS =  vec4(((vec2(tid)+0.5)/pushConstants.view_size)*2-1.0f, sampleDepth, 1);
    rayStartInfo.samplePosInCS.y *= -1;
    // From the depth, compute the position in view space.
    rayStartInfo.samplePosInVS = pushConstants.InvProjMat * rayStartInfo.samplePosInCS;
    rayStartInfo.samplePosInVS /= rayStartInfo.samplePosInVS.w;
    // Texture space
    rayStartInfo.samplePosInTS = rayStartInfo.samplePosInCS.xyz;
    rayStartInfo.samplePosInTS.xy *= vec2(0.5f, -0.5f);
    rayStartInfo.samplePosInTS.xy += vec2(0.5f, 0.5f);
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

    vec4 samplePosInCS =  vec4(((vec2(tid)+0.5)/pushConstants.view_size)*2-1.0f, sampleDepth, 1);
    samplePosInCS.y *= -1;

    vec4 samplePosInVS = pushConstants.InvProjMat * samplePosInCS;
    samplePosInVS /= samplePosInVS.w;

    vec3 vCamToSampleInVS = normalize(samplePosInVS.xyz);
    vec4 vReflectionInVS = vec4(reflect(vCamToSampleInVS.xyz, vSampleNormalInVS.xyz),0);

    vec4 vReflectionEndPosInVS = samplePosInVS + vReflectionInVS * 1000;
    vReflectionEndPosInVS /= (vReflectionEndPosInVS.z < 0 ? vReflectionEndPosInVS.z : 1);
    vec4 vReflectionEndPosInCS = pushConstants.ProjMat * vec4(vReflectionEndPosInVS.xyz, 1);
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
    // vReflectionEndPosInVS /= (vReflectionEndPosInVS.z < 0 ? vReflectionEndPosInVS.z : 1);
    vec4 vReflectionEndPosInCS = pushConstants.ProjMat * vec4(vReflectionEndPosInVS.xyz, 1);
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