

ivec2 sampleHiLumin(inout uint RNG, out float pdf) {
    int mip_level = pushConstants.hilum_mip_levels - 1;
    ivec2 xy = ivec2(0, 0);
    float parent_importance = texelFetch(hi_luminance, xy, mip_level).x;
    
    ivec2 four_neighbors[4];
    four_neighbors[0] = ivec2(0, 0);
    four_neighbors[1] = ivec2(0, 1);
    four_neighbors[2] = ivec2(1, 0);
    four_neighbors[3] = ivec2(1, 1);

    float p = 1.f;
    
    for (int i = maximum_lod-1; i>=0; --i) {
        if(i!=maximum_lod-1) {
            xy *= 2;
        }
        mip_level--; // next mip level
        const float rnd = UniformFloat(RNG);  // sample next level
        float accum = 0.f;
        int last_non_zero = -1;
        float last_non_zero_imp = 0;
        float last_non_zero_pdf = 0;
        for (int j=0; j<4; ++j) {
            ivec2 xy_offset = four_neighbors[j];
            float importance = texelFetch(boostrapImportMIP, xy + xy_offset, mip_level).x;
            float nimportance = importance / parent_importance;
            accum += nimportance;
            
            if(nimportance > 0) {
                last_non_zero = j;
                last_non_zero_pdf = nimportance;
                last_non_zero_imp = importance;
            }

            if(rnd < accum) {
                xy = xy + xy_offset;
                p = p * nimportance;
                parent_importance = importance;
                break;
            }
            else if(j==3 && last_non_zero!=-1) {
                xy = xy + four_neighbors[last_non_zero];
                p = p * last_non_zero_pdf;
                parent_importance = last_non_zero_imp;
                break;
            }
            else {
                // should not happen...
            }
        }
    }
    pdf = p;
    return xy;
}