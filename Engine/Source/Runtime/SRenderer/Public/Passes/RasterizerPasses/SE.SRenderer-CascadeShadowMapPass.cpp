#include "SE.SRenderer-CascadeShadowMapPass.hpp"

namespace SIByL {
auto fitToSceneZBound(Math::bounds3 const& aabbBounds, Math::mat4 w2l) noexcept
    -> Math::vec2 {
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
  Math::vec3 localPosition =
      Math::vec3((boundsLocal.pMax.x + boundsLocal.pMin.x) / 2,
                 (boundsLocal.pMax.y + boundsLocal.pMin.y) / 2, 0);
  Math::vec2 orthographicSize = std::max(xsize, ysize);
  float const nearClipPlane = boundsLocal.pMin.z;
  float const farClipPlane = boundsLocal.pMax.z;
  return Math::vec2(nearClipPlane, farClipPlane);
}

std::vector<Math::vec4> getFrustumCornersWorldSpace(Math::mat4 const& proj,
                                                    Math::mat4 const& view) {
  const auto inv = Math::inverse(proj * view);
  std::vector<Math::vec4> frustumCorners;
  for (unsigned int x = 0; x < 2; ++x) {
    for (unsigned int y = 0; y < 2; ++y) {
      for (unsigned int z = 0; z < 2; ++z) {
        const Math::vec4 pt =
            inv * Math::vec4(2.0f * x - 1.0f, 2.0f * y - 1.0f, z, 1.0f);
        frustumCorners.push_back(pt / pt.w);
      }
    }
  }
  return frustumCorners;
}

Math::mat4 getLightSpaceMatrix(Math::mat4 const& cam_view,
                               Math::vec3 const& lightDir,
                               const float nearPlane, const float farPlane,
                               Math::bounds3 bounds,
                               Math::Transform const& w2l_transform) {
  GFX::CameraComponent const* camera = RACommon::get()->mainCamera;
  const auto proj =
      Math::perspective(camera->fovy, camera->aspect, nearPlane, farPlane).m;
  const auto corners = getFrustumCornersWorldSpace(proj, cam_view);
  const auto inv = Math::inverse(proj * cam_view);
  // Math::vec3 center = Math::vec3(0, 0, 0);
  // for (const auto& v : corners) {
  //	center += Math::vec3(v.x, v.y, v.z);
  // }
  // center /= (float)(corners.size());

  float minX = std::numeric_limits<float>::max();
  float maxX = std::numeric_limits<float>::lowest();
  float minY = std::numeric_limits<float>::max();
  float maxY = std::numeric_limits<float>::lowest();
  float minZ = std::numeric_limits<float>::max();
  float maxZ = std::numeric_limits<float>::lowest();

  for (const auto& v : corners) {
    const auto trf = w2l_transform.m * v;
    minX = std::min(minX, trf.x);
    maxX = std::max(maxX, trf.x);
    minY = std::min(minY, trf.y);
    maxY = std::max(maxY, trf.y);
    minZ = std::min(minZ, trf.z);
    maxZ = std::max(maxZ, trf.z);
  }

  minX = std::max(minX, bounds.pMin.x);
  maxX = std::min(maxX, bounds.pMax.x);
  minY = std::max(minY, bounds.pMin.y);
  maxY = std::min(maxY, bounds.pMax.y);
  // minX = bounds.pMin.x;
  // maxX = bounds.pMax.x;
  // minY = bounds.pMin.y;
  // maxY = bounds.pMax.y;
  // minZ = bounds.pMin.z;
  // maxZ = bounds.pMax.z;

  // Tune this parameter according to the scene
  minZ = std::min(bounds.pMin.z, minZ);
  maxZ = std::min(bounds.pMax.z, maxZ);

  float padding_x = (maxX - minX) * 0.05;
  float padding_y = (maxY - minY) * 0.05;
  float padding_z = (maxZ - minZ) * 0.05;

  Math::mat4 const lightProjection =
      Math::ortho(std::floor(minX - padding_x), std::ceil(maxX + padding_x),
                  std::floor(minY - padding_y), std::ceil(maxY + padding_y),
                  minZ - padding_z, maxZ + padding_z)
          .m;

  return lightProjection * w2l_transform.m;
}

std::vector<Math::mat4> getLightSpaceMatrices() {
  // find real near / far considering scene AABB
  Math::mat4 cam_view = RACommon::get()->mainCameraInfo.view;
  auto const& aabbBounds = RACommon::get()->sceneAABB;
  GFX::CameraComponent const* camera = RACommon::get()->mainCamera;
  float z_near = std::numeric_limits<float>::max();
  float z_far = std::numeric_limits<float>::min();
  for (size_t i = 0; i < 8; ++i) {
    Math::vec4 pointInViewSpace =
        cam_view * Math::vec4(Math::vec3(aabbBounds.corner(i)), 1);
    z_far = std::max(pointInViewSpace.z, z_far);
    z_near = std::min(pointInViewSpace.z, z_near);
  }
  z_far = std::min(z_far, camera->far);
  z_near = std::max(z_near, camera->near);

  // declare shadow map levels
  const float z_step = z_far - z_near;
  std::vector<float> shadowCascadeLevels{
      z_near + z_step / 4, z_near + z_step * 2 / 4, z_near + z_step * 3 / 4};

  Math::mat4 transform = RACommon::get()->mainDirectionalLight->transform;
  Math::vec3 lightDir = Math::Transform(transform) * Math::vec3(0, 0, 1);
  lightDir = Math::normalize(lightDir);
  Math::mat4 w2l = Math::lookAt(Math::vec3{0}, lightDir, Math::vec3{0, 1, 0}).m;

  RACommon::get()->cascade_distances =
      Math::vec4{z_near + z_step / 4, z_near + z_step * 2 / 4,
                 z_near + z_step * 3 / 4, z_far};

  // see the z-bounds
  // Create bounds in light space
  Math::bounds3 boundsLocal = {};
  Math::Transform w2l_transform(w2l);
  for (size_t i = 0; i < 8; ++i) {
    Math::point3 pointInLightSpace = w2l_transform * aabbBounds.corner(i);
    boundsLocal = Math::unionPoint(boundsLocal, pointInLightSpace);
  }
  float zsize = (boundsLocal.pMax.z - boundsLocal.pMin.z) / 2;
  float const nearClipPlane = boundsLocal.pMin.z;
  float const farClipPlane = boundsLocal.pMax.z;
  Math::vec2 zbound = {nearClipPlane, farClipPlane};

  std::vector<Math::mat4> ret;
  for (size_t i = 0; i < shadowCascadeLevels.size() + 1; ++i) {
    if (i == 0) {
      ret.push_back(getLightSpaceMatrix(cam_view, lightDir, z_near,
                                        shadowCascadeLevels[i], boundsLocal,
                                        w2l_transform));
    } else if (i < shadowCascadeLevels.size()) {
      ret.push_back(getLightSpaceMatrix(
          cam_view, lightDir, shadowCascadeLevels[i - 1],
          shadowCascadeLevels[i], boundsLocal, w2l_transform));
    } else {
      ret.push_back(getLightSpaceMatrix(cam_view, lightDir,
                                        shadowCascadeLevels[i - 1], z_far,
                                        boundsLocal, w2l_transform));
    }
  }
  return ret;
}
}  // namespace SIByL