#include "SE.SRenderer-ShadowMapPass.hpp"

namespace SIByL {
SE_EXPORT auto fitToScene(Math::bounds3 const& aabbBounds,
                          Math::mat4 w2l) noexcept -> Math::mat4 {
  // Create bounds in light space
  Math::bounds3 boundsLocal = {};
  Math::Transform w2l_transform(w2l);
  for (size_t i = 0; i < 8; ++i) {
    Math::point3 pointInLightSpace = w2l_transform * aabbBounds.corner(i);
    boundsLocal = Math::unionPoint(boundsLocal, pointInLightSpace);
  }
  float xsize = (boundsLocal.pMax.x - boundsLocal.pMin.x) / 2;
  float ysize = (boundsLocal.pMax.y - boundsLocal.pMin.y) / 2;
  float zsize = (boundsLocal.pMax.z - boundsLocal.pMin.z) / 2;
  // Create camera info
  Math::vec3 const localPosition =
      Math::vec3((boundsLocal.pMax.x + boundsLocal.pMin.x) / 2,
                 (boundsLocal.pMax.y + boundsLocal.pMin.y) / 2, 0);
  Math::vec2 const orthographicSize = std::max(xsize, ysize);
  float const nearClipPlane = boundsLocal.pMin.z;
  float const farClipPlane = boundsLocal.pMax.z;
  Math::vec2 const center =
      Math::vec2(0.5f * (boundsLocal.pMax.x + boundsLocal.pMin.x),
                 0.5f * (boundsLocal.pMax.y + boundsLocal.pMin.y));
  return Math::ortho(center.x - orthographicSize.x,
                     center.x + orthographicSize.x,
                     center.y - orthographicSize.y,
                     center.y + orthographicSize.y, nearClipPlane, farClipPlane)
      .m;
}
}  // namespace SIByL