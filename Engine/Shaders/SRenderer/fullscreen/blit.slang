Sampler2D<RESOURCE_TYPE> source;

[shader("fragment")]
RESOURCE_TYPE fragmentMain(
    in float2 uv: Coord) : SV_Target
{
    return source.Sample(uv);
}