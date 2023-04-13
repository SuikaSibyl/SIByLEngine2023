
vec2 saturate(vec2 v) { return clamp(v, 0.0, 1.0); }

// Returns the number of cells in the quad tree at the given level.
vec2 getCellCount(in const int mipLevel) {
    return textureSize(hi_z, mipLevel);
}

// Returns the 2D integer index of the cell that contains the given 2D position within it.
vec2 getCell(in const vec2 pos, in const vec2 cell_count) {
    return vec2(floor(pos * cell_count));
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


vec3 intersect_cell_boundary(vec3 pos, vec3 dir, vec2 cell_id, vec2 cell_count, vec2 cross_step, vec2 cross_offset) {
    vec2 cell_size = 1.0 / cell_count;
    vec2 planes = cell_id / cell_count + cell_size * cross_step;

    vec2 solutions = (planes - pos.xy) / dir.xy;
    vec3 intersection_pos = pos + dir * min(solutions.x, solutions.y);

    intersection_pos.xy += (solutions.x < solutions.y) ? vec2(cross_offset.x, 0.0) : vec2(0.0, cross_offset.y);

    return intersection_pos;
}

bool crossed_cell_boundary(vec2 cell_id_one, vec2 cell_id_two) {
    return int(cell_id_one.x) != int(cell_id_two.x) || int(cell_id_one.y) != int(cell_id_two.y);
}

float minimumDepthPlane(vec2 ray, int level, vec2 cell_count) {
    return texelFetch(hi_z, ivec2(ray.xy * cell_count), level).r;
}

bool FindIntersection_HiZ(
    vec3 samplePosInTS,
    vec3 vReflDirInTS,
    float maxTraceDistance,
    out vec3 intersection
) {
    const int maxLevel = pushConstants.hiz_mip_levels - 1;
	
    vec2 crossStep = vec2(vReflDirInTS.x>=0 ? 1 : -1, vReflDirInTS.y>=0 ? 1 : -1);
    vec2 crossOffset = crossStep / ivec2(pushConstants.view_size) / 128;
    crossStep = saturate(crossStep);
    
    vec3 ray = samplePosInTS.xyz;
    float minZ = ray.z;
    float maxZ = ray.z + vReflDirInTS.z * maxTraceDistance;
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

    if(isBackwardRay) return false;
    
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
            if((cell_minZ > ray.z) ){
                intersection = tmpRay;
                return true;
            }
            else {
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

bool FindIntersection_Linear(
    in const vec3 samplePosInTS,
    in const vec3 vReflDirInTS,
    in const float maxTraceDistance,
    out vec3 intersection
) {
    const vec3 vReflectionEndPosInTS = samplePosInTS + vReflDirInTS * maxTraceDistance;
    // dp is the vector that moves the current ray to its next position by adding it to the current position of the ray.
    vec3 dp = vReflectionEndPosInTS.xyz - samplePosInTS.xyz;
    ivec2 sampleScreenPos = ivec2(samplePosInTS.xy * pushConstants.view_size.xy);
    ivec2 endPosScreenPos = ivec2(vReflectionEndPosInTS.xy * pushConstants.view_size.xy);
    ivec2 dp2 = endPosScreenPos - sampleScreenPos;
    const int max_dist = max(abs(dp2.x), abs(dp2.y));
    dp /= max_dist;
    
    vec4 rayPosInTS = vec4(samplePosInTS.xyz + dp, 0);
    vec4 vRayDirInTS = vec4(dp.xyz, 0);
	vec4 rayStartPos = rayPosInTS;

    int hitIndex = -1;
    for(int i = 0;i<=max_dist && i<pushConstants.max_iteration; i += 4)  {
        float depth0 = 0;
        float depth1 = 0;
        float depth2 = 0;
        float depth3 = 0;

        vec4 rayPosInTS0 = rayPosInTS+vRayDirInTS*0;
        vec4 rayPosInTS1 = rayPosInTS+vRayDirInTS*1;
        vec4 rayPosInTS2 = rayPosInTS+vRayDirInTS*2;
        vec4 rayPosInTS3 = rayPosInTS+vRayDirInTS*3;

        depth3 = texture(hi_z, rayPosInTS3.xy).x;
        depth2 = texture(hi_z, rayPosInTS2.xy).x;
        depth1 = texture(hi_z, rayPosInTS1.xy).x;
        depth0 = texture(hi_z, rayPosInTS0.xy).x;

        {   float thickness = rayPosInTS3.z - depth3;
            hitIndex = (thickness>=0 && thickness < pushConstants.max_thickness) ? (i+3) : hitIndex;
        } {
            float thickness = rayPosInTS2.z - depth2;
            hitIndex = (thickness>=0 && thickness < pushConstants.max_thickness) ? (i+2) : hitIndex;
        } {
            float thickness = rayPosInTS1.z - depth1;
            hitIndex = (thickness>=0 && thickness < pushConstants.max_thickness) ? (i+1) : hitIndex;
        } {
            float thickness = rayPosInTS0.z - depth0;
            hitIndex = (thickness>=0 && thickness < pushConstants.max_thickness) ? (i+0) : hitIndex;
        }

        if(hitIndex != -1) break;

        rayPosInTS = rayPosInTS3 + vRayDirInTS;
    }

    bool intersected = hitIndex >= 0;
    intersection = rayStartPos.xyz + vRayDirInTS.xyz * hitIndex;

    const vec2 cellCount = getCellCount(0); 
    const vec2 oldCellIdx = getCell(intersection.xy, cellCount);    
    return intersected;
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