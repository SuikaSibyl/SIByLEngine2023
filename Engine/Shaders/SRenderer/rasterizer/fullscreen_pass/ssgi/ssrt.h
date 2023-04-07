

vec2 getCellCount(int mipLevel) {
    return vec2(tex_hi_z.get_width(mipLevel) ,tex_hi_z.get_height(mipLevel));
}


/**
* Hi-Z ray tracing algorithm
* @ref: https://sugulee.wordpress.com/2021/01/19/screen-space-reflections-implementation-and-optimization-part-2-hi-z-tracing-method/
*/
float FindIntersection_HiZ(
    in const vec3 samplePosInTS,
    in const vec3 vReflDirInTS,
    in const float maxTraceDistance,
    in const uint nMipLevels,
    out vec3 intersection)
{
    // sets maxLevel which is just the level of the last mip in the Hi-Z texture
    const int maxLevel = nMipLevels - 1;
	
    // Set crossStep & crossOffset
    // These two variables are used to move the ray to the next cell in the grid of the quad tree
    vec2 crossStep = vec2(vReflDirInTS.x>=0 ? 1 : -1, vReflDirInTS.y>=0 ? 1 : -1);
    vec2 crossOffset = crossStep / sceneInfo.ViewSize / 128;
    crossStep = saturate(crossStep);
    
    vec3 ray = samplePosInTS.xyz;
    float minZ = ray.z;
    float maxZ = ray.z + vReflDirInTS.z * maxTraceDistance;
    float deltaZ = (maxZ - minZ);

    vec3 o = ray;
    vec3 d = vReflDirInTS * maxTraceDistance;
	
    int startLevel = 2;
    int stopLevel = 0;
    vec2 startCellCount = getCellCount(startLevel, tex_hi_z);
	
    vec2 rayCell = getCell(ray.xy, startCellCount);
    ray = intersectCellBoundary(o, d, rayCell, startCellCount, crossStep, crossOffset*64);
    
    int level = startLevel;
    uint iter = 0;
    bool isBackwardRay = vReflDirInTS.z<0;
    float rayDir = isBackwardRay ? -1 : 1;

    while(level >=stopLevel && ray.z*rayDir <= maxZ*rayDir && iter<sceneInfo.maxIteration) {
        
        const vec2 cellCount = getCellCount(level, tex_hi_z);
        const vec2 oldCellIdx = getCell(ray.xy, cellCount);
        
        float cell_minZ = getMinimumDepthPlane((oldCellIdx+0.5f)/cellCount, level, tex_hi_z);
        vec3 tmpRay = ((cell_minZ > ray.z) && !isBackwardRay) ? intersectDepthPlane(o, d, (cell_minZ - minZ)/deltaZ) : ray;
        
        const vec2 newCellIdx = getCell(tmpRay.xy, cellCount);
        
        float thickness = level == 0 ? (ray.z - cell_minZ) : 0;
        bool crossed = (isBackwardRay && (cell_minZ > ray.z)) || (thickness>(MAX_THICKNESS)) || crossedCellBoundary(oldCellIdx, newCellIdx);
        ray = crossed ? intersectCellBoundary(o, d, oldCellIdx, cellCount, crossStep, crossOffset) : tmpRay;
        level = crossed ? min((float)maxLevel, level + 1.0f) : level-1;
        
        ++iter;
    }
    
    bool intersected = (level < stopLevel);
    intersection = ray;
	
    float intensity = intersected ? 1 : 0;
    
    return intensity;
}
