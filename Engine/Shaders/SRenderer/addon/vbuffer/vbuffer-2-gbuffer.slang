#include "vbuffer.hlsli"
#include "../gbuffer/gbuffer_common.hlsli"

const Texture2D<uint4> u_vBuffer;

RWTexture2D<float4>  u_GBufferPosition;
RWTexture2D<uint>   u_GBufferNormals;
RWTexture2D<uint>   u_GBufferGeoNormals;
RWTexture2D<uint>   u_GBufferDiffuseAlbedo;
RWTexture2D<uint>   u_GBufferSpecularRough;
RWTexture2D<float4> u_Emissive;
RWTexture2D<float4> u_MotionVectors;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    const int2 pixel = DTid.xy;
    const int2 resolution = getViewportSize(globalUniform.cameraData);
    if (any(pixel >= resolution)) return;

    // output data
    float viewDepth = k_background_depth;
    float3 motion = float3(0);
    float3 diffuseAlbedo = float3(0);
    float3 specularF0 = float3(0);
    float3 flatNormal = float3(0);
    float3 shadingNormal = float3(0);
    float3 emission = float3(0);
    float3 positionWS = float3(0);
    float roughness = 1.f;
    float viewDistance = 0;
    float fwidth_depth = 0;
    
    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );

    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);

    if (!IsInvalid(vhit)) {
        const int geometryID = int(vhit.geometryID);
        const int primitiveID = int(vhit.primitiveID);
        const float3 bary = float3(1 - vhit.barycentric.x - vhit.barycentric.y, vhit.barycentric.xy);

        const GeometryInfo geometry = geometries[geometryID];
        const GeometryInfo prevGeometry = prevGeometries[geometryID];

        const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));
        const MaterialInfo material = materials[geometry.materialID];

        float3 vertexPositions[3];
        vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
        vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
        vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));

        const float4x4 o2w = ObjectToWorld(geometry);
        const float3 positionOS = interpolate(vertexPositions, bary);
        positionWS = mul(float4(positionOS, 1.0), o2w).xyz;

        float3 viewDirection = positionWS - globalUniform.cameraData.posW;
        viewDistance = length(viewDirection);
        viewDirection /= viewDistance;

        float2 vertexUVs[3];
        vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
        vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
        vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
        float2 uv = interpolate(vertexUVs, bary);

        float3 objectSpaceFlatNormal = normalize(cross(
            vertexPositions[1] - vertexPositions[0],
            vertexPositions[2] - vertexPositions[0]));

        float4x4 o2wn = ObjectToWorldNormal(geometry);
        flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);
        bool faceforward = dot(flatNormal, viewDirection) < 0;
        flatNormal = faceforward ? flatNormal : -flatNormal;

        float3 normals[3];
        normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
        normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
        normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
        float3 vertexNormalOS = interpolate(normals, bary);
        float3 vertexNormalWS = normalize(mul(float4(vertexNormalOS, 0.0), o2wn).xyz);
        shadingNormal = faceforward ? vertexNormalWS : -vertexNormalWS;
        
        float3 tangents[3];
        tangents[0] = fetchVertexTangent(index[0] + int(geometry.vertexOffset));
        tangents[1] = fetchVertexTangent(index[1] + int(geometry.vertexOffset));
        tangents[2] = fetchVertexTangent(index[2] + int(geometry.vertexOffset));
        float3 tangentOS = interpolate(tangents, bary);
        float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), geometry.oddNegativeScaling);

        // Grab material infomation
        diffuseAlbedo = textures[material.baseOrDiffuseTextureIndex].Sample(uv).rgb;

        motion = getMotionVector(globalUniform.cameraData,
                                 prevGlobalUniform.cameraData,
                                 geometry,
                                 prevGeometry,
                                 positionOS,
                                 positionOS, // no vertex change
                                 viewDepth);

        specularF0 = float3(0);
        roughness = 1;

        const float2 inUV = float2(pixel);
        const Ray ray_0 = generateRay(inUV, resolution, false, globalUniform.cameraData);
        const Ray ray_x = generateRay(inUV + float2(1, 0), resolution, false, globalUniform.cameraData);
        const Ray ray_y = generateRay(inUV + float2(0, 1), resolution, false, globalUniform.cameraData);
        const float half_cone_angle = sqrt(1.0 - square(min(dot(ray_0.direction, ray_x.direction), dot(ray_0.direction, ray_y.direction))));
        fwidth_depth = computeFWidthDepth(viewDistance * half_cone_angle, viewDirection, flatNormal);
    }
    
    // write to gbuffer
    u_GBufferPosition[pixel] = float4(positionWS, viewDepth);
    u_GBufferDiffuseAlbedo[pixel] = Pack_R11G11B10_UFLOAT(diffuseAlbedo);
    u_GBufferSpecularRough[pixel] = Pack_R8G8B8A8_Gamma_UFLOAT(float4(specularF0, roughness));
    u_GBufferNormals[pixel] = UnitVectorToUnorm32Octahedron(shadingNormal);
    u_GBufferGeoNormals[pixel] = UnitVectorToUnorm32Octahedron(flatNormal);
    u_Emissive[pixel] = float4(emission, viewDistance);
    u_MotionVectors[pixel] = float4(motion, fwidth_depth);
}