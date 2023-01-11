#include "../include/common_hit.h"
#include "../include/common_sample_shape.h"
#include "../../../Utility/random.h"

layout(location = 0) rayPayloadInEXT PrimaryPayload primaryPld;

void main()
{
    // set primary payload
    HitGeometry geoInfo = getHitGeometry();
    primaryPld.position         = geoInfo.worldPosition;
    primaryPld.geometryNormal   = geoInfo.geometryNormal;
    primaryPld.TBN              = geoInfo.TBN;
    primaryPld.matID            = geoInfo.matID;
    primaryPld.uv               = geoInfo.uv;
    setIntersected(primaryPld.flags, true);
}