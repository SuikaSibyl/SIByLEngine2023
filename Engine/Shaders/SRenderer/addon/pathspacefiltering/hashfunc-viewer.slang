#include "../../include/common/hashing.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 iResolution;
    float iTime;
};

float2 to_shadertoy_uv(float2 uv) {
    return float2(uv.x, 1 - uv.y);
}

float2 to_shadertoy_fragcoord(float2 uv) {
    return uv * iResolution;
}

uint3 hash(float2 s) {
    /*	Uncomment the hash you want to visualize.

        Note that most of these examples the hash is only given 2 inputs unless the hash requires more.
        But Common includes 1 through 4 input variations of the hash if they exist.

        You should play around with different seeds/different number of inputs.

        Available hashes:

        bbs, city, esgtsa, fast, hashwithoutsine, hybridtaus,
        ign, iqint1, iqint2, iqint3, jkiss32, lcg, md5, murmur3,
        pcg, pcg2d, pcg3d, pcg3d16, pcg4d, pseudo, ranlim32,
        superfast, tea2, tea3, tea4, tea5, trig, wang,
        xorshift128, xorshift32, xxhash32
    */

    uint4 u = uint4(uint2(s), uint(s.x) ^ uint(s.y), uint(s.x) + uint(s.y)); // Play with different values for 3rd and 4th params. Some hashes are okay with constants, most aren't.

    // return uvec3(bbs(seed(u.xy)));
    // return uvec3(bbs(bbs(u.x) + u.y));
    // return uvec3(city(u.xy));
    // return uvec3(esgtsa(seed(u.xy)));
    // return uvec3(esgtsa(esgtsa(u.x) + u.y));
    // return uvec3(fast(s) * float(0xffffffffu));
    // return uvec3(hashwithoutsine32(s) * float(0xffffffffu));
    // return uvec3(hybridtaus(u));
    // return uvec3(ign(s) * float(0xffffffffu));
    // return uvec3(iqint1(seed(u.xy)));
    // return uvec3(iqint1(iqint1(u.x) + u.y));
    // return iqint2(u.xyz);
    // return uvec3(iqint3(u.xy));
    // return uvec3(jkiss32(u.xy));
    // return uvec3(lcg(seed(u.xy)));
    // return uvec3(lcg(lcg(u.x) + u.y));
    // return md5(u).xyz;
    // return uvec3(murmur3(u.xy));
    // return uvec3(pcg(seed(u.xy)));
    // return uvec3(pcg(pcg(u.x) + u.y));
    // return uvec3(pcg2d(u.xy), 0u);
    return pcg3d(u.xyz);
    // return pcg3d16(u.xyz);
    // return pcg4d(u).xyz;
    // return uvec3(pseudo(s) * float(0xffffffffu));
    // return uvec3(ranlim32(seed(u.xy)));
    // return uvec3(ranlim32(ranlim32(u.x) + u.y));
    // return uvec3(superfast(seed(u.xy)));
    // return uvec3(superfast(superfast(u.x) + u.y));
    // return uvec3(tea(2, u.xy), 0u);
    // return uvec3(tea(3, u.xy), 0u);
    // return uvec3(tea(4, u.xy), 0u);
    // return uvec3(tea(5, u.xy), 0u);
    // return uvec3(trig(s) * float(0xffffffffu));
    // return uvec3(wang(seed(u.xy)));
    // return uvec3(wang(wang(u.x) + u.y));
    // return uvec3(xorshift128(u));
    // return uvec3(xorshift32(seed(u.xy)));
    // return uvec3(xorshift32(xorshift32(u.x) + u.y));
    // return uvec3(xxhash32(u.xy));
}

[shader("fragment")]
float4 fragmentMain(in float2 uv: Coord) : SV_Target {
    uv = to_shadertoy_uv(uv);
    const float2 fragCoord = to_shadertoy_fragcoord(uv);

    float4 fragColor = float4(uv, 0, 1);
    const uint bit = uint(8.0 * uv.x) + 8u * uint(4.0 * uv.y);
    float2 seed = fmod(fragCoord, float2(iResolution.x / 8.0, iResolution.y / 4.0));
    
    seed += 100.0 * iTime;

    uint3 hash = hash(seed);

    fragColor = float4((hash >> bit) & 1u, 1.0);

    fragColor *= step(10.0 / iResolution.x, 1.0 - abs(2.0 * frac(8.0 * uv.x) - 1.0));
    fragColor *= step(10.0 / iResolution.y, 1.0 - abs(2.0 * frac(4.0 * uv.y) - 1.0));
    
    return fragColor;
}