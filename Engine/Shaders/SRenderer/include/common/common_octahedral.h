#ifndef _SRENDERER_COMMON_OCTAHEDRAL_INCLUDED_
#define _SRENDERER_COMMON_OCTAHEDRAL_INCLUDED_

/**
* Octahedral map functions
* @ref: adaptive from UnrealEngine-5.0.3-release & ShaderToy
* @url: UnrealEngine-5.0.3-release/Engine/Shaders/Private/OctahedralCommon.ush
* @url: https://www.shadertoy.com/view/flcXRl
*/

/** Helper function to reflect the folds of the lower hemisphere
* over the diagonals in the octahedral map.
* As xy could be 0, using sign could return 0, so we use the mix. */
vec2 octWrap(in const vec2 v) {
    return (1.f - abs(v.yx)) * mix(vec2(1), vec2(-1), lessThan(v.xy, vec2(0)));
}

/** Signed encodings
* Converts a normalized direction to the octahedral map (non-equal area, signed)
* Returns a signed position in octahedral map [-1, 1] for each component
* @param normal: normalized direction */
vec2 UnitVectorToSignedOctahedron(vec3 normal) {
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1)
	normal.xy /= dot(vec3(1), abs(normal));
	// Then project the octahedron onto the xy-plane
	return (normal.z < 0) ? octWrap(normal.xy) : normal.xy;
}

/** Signed decodings
* Converts a point in the octahedral map to a normalized direction (non-equal area, signed)
* @param oct: signed position in octahedral map [-1, 1] for each component
* @return normalized direction */
vec3 SignedOctahedronToUnitVector(vec2 oct) {
	vec3 normal = vec3(oct, 1 - dot(vec2(1), abs(oct)));
	const float t = max(-normal.z, 0);
	normal.xy += mix(vec2(-t), vec2(t), lessThan(normal.xy, vec2(0)));
	return normalize(normal);
}

/** Signed encodings (hemisphere)
* Converts a normalized direction to the hemisphere octahedral map (non-equal area, signed)
* Returns a signed position in octahedral map [-1, 1] for each component
* @param normal: normalized direction, but the z component would lost */
vec2 UnitVectorToHemiOctahedron(vec3 normal) {
	normal.xy /= dot(vec3(1), abs(normal));
	return vec2(normal.x + normal.y, normal.x - normal.y);
}

/** Signed decodings (hemisphere)
* Converts a point in the octahedral map to a normalized direction (non-equal area, signed)
* @param oct: signed position in octahedral map [-1, 1] for each component
* @return normalized direction, assume the z component is always positive. */
vec3 HemiOctahedronToUnitVector(vec2 oct) {
	oct = vec2(oct.x + oct.y, oct.x - oct.y);
	vec3 N = vec3(oct, 2.0 - dot(vec2(1), abs(oct)));
	return normalize(N);
}

/** Unorm 32 bit encodings
* Converts a normalized direction to the octahedral map (non-equal area, unsigned normalized)
* @param normal: normalized direction.
* @return packed 32 bit unsigned normalized position in octahedral map.
* 		  The two components of the result are stored in UNORM16 format, [0..1] */
uint UnitVectorToUnorm32Octahedron(vec3 normal) {
    vec2 p = UnitVectorToSignedOctahedron(normal);
    p = clamp(p.xy * 0.5 + 0.5, 0., 1.);
    return uint(p.x * 0xfffe) | (uint(p.y * 0xfffe) << 16);
}

/** Unorm 32 bit decodings
* Converts a point in the octahedral map (non-equal area, unsigned normalized) to normalized direction
* @param pNorm: a packed 32 bit unsigned normalized position in octahedral map
* @return normalized direction */
vec3 Unorm32OctahedronToUnitVector(uint pUnorm) {
    vec2 p;
    p.x = clamp(float(pUnorm & 0xffff) / 0xfffe, 0., 1.);
    p.y = clamp(float(pUnorm >> 16) / 0xfffe, 0., 1.);
    p = p * 2.0 - 1.0;
    return SignedOctahedronToUnitVector(p);
}

// Wrap around octahedral map for correct hardware bilinear filtering
uvec2 OctahedralMapWrapBorder(uvec2 TexelCoord, uint Resolution, uint BorderSize) {
	if (TexelCoord.x < BorderSize) {
		TexelCoord.x = BorderSize - 1 + BorderSize - TexelCoord.x;
		TexelCoord.y = Resolution - 1 - TexelCoord.y;
	}
	if (TexelCoord.x >= Resolution - BorderSize) {
		TexelCoord.x = (Resolution - BorderSize) - (TexelCoord.x - (Resolution - BorderSize - 1));
		TexelCoord.y = Resolution - 1 - TexelCoord.y;
	}
	if (TexelCoord.y < BorderSize) {
		TexelCoord.y = BorderSize - 1 + BorderSize - TexelCoord.y;
		TexelCoord.x = Resolution - 1 - TexelCoord.x;
	}
	if (TexelCoord.y >= Resolution - BorderSize) {
		TexelCoord.y = (Resolution - BorderSize) - (TexelCoord.y - (Resolution - BorderSize - 1));
		TexelCoord.x = Resolution - 1 - TexelCoord.x;
	}
	return TexelCoord - BorderSize;
}

// Computes the spherical excess (solid angle) of a spherical triangle with vertices A, B, C as unit length vectors
// https://en.wikipedia.org/wiki/Spherical_trigonometry#Area_and_spherical_excess
float ComputeSphericalExcess(vec3 A, vec3 B, vec3 C) {
    const float CosAB = dot(A, B);
    const float SinAB = 1.0f - CosAB * CosAB;
    const float CosBC = dot(B, C);
    const float SinBC = 1.0f - CosBC * CosBC;
    const float CosCA = dot(C, A);
    const float CosC = CosCA - CosAB * CosBC;
    const float SinC = sqrt(SinAB * SinBC - CosC * CosC);
    const float Inv = (1.0f - CosAB) * (1.0f - CosBC);
	return 2.0f * atan2(SinC, sqrt((SinAB * SinBC * (1.0f + CosBC) * (1.0f + CosAB)) / Inv) + CosC);
}

// Notice that the octahedral solid angle are different in each texel.
// TexelCoord should be centered on the octahedral texel, in the range [.5f, .5f + Resolution - 1]
float OctahedralSolidAngle(vec2 TexelCoord, float InvResolution) {
	vec3 Direction10 = SignedOctahedronToUnitVector(TexelCoord + vec2(.5f, -.5f) * InvResolution);
	vec3 Direction01 = SignedOctahedronToUnitVector(TexelCoord + vec2(-.5f, .5f) * InvResolution);

	float SolidAngle0 = ComputeSphericalExcess(
		SignedOctahedronToUnitVector(TexelCoord + vec2(-.5f, -.5f) * InvResolution), 
		Direction10,
		Direction01);

	float SolidAngle1 = ComputeSphericalExcess(
		SignedOctahedronToUnitVector(TexelCoord + vec2(.5f, .5f) * InvResolution), 
		Direction01,
		Direction10);

	return SolidAngle0 + SolidAngle1;
}

#endif