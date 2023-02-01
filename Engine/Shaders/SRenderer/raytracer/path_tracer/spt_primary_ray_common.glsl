#include "../include/common_hit.h"
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
    primaryPld.geometryID       = geoInfo.geometryID;
    primaryPld.lightID          = geoInfo.lightID;
    primaryPld.hitFrontface     = (geoInfo.geometryNormalUnflipped == geoInfo.geometryNormal) ? 1.f : -1.f;
#if PRIMITIVE_TYPE == PRIMITIVE_SPHERE
    primaryPld.normalFlipping   = 0.f;
#elif PRIMITIVE_TYPE == PRIMITIVE_TRIANGLE
    primaryPld.normalFlipping   = primaryPld.hitFrontface;
#endif
    setIntersected(primaryPld.flags, true);
}