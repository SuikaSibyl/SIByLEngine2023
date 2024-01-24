#include "../Public/SE.GFX-SceneUpdate.h"
#include <stack>

namespace SIByL::GFX {
auto update_animation(GFX::Scene& scene) noexcept -> void {
  auto view = Core::ComponentManager::get()->view
      <GFX::TransformComponent, GFX::AnimationComponent>();
  float sec = GFX::GFXManager::get()->mTimeline.currentSec;
  // sec *= 2;
  // sec *= 2;

  {
    auto temp_view = Core::ComponentManager::get()
        ->view<GFX::TransformComponent, GFX::TagComponent>();
    //for (auto& [entity, transform, tag] : temp_view) {
    //  if (tag.name == "DirectionalLight") {
    //    transform.eulerAngles.z = 25 * std::cos(sec * 1.6);
    //  }
    //  if (tag.name == "Motor001") {
    //    transform.eulerAngles.x = -89.9;
    //    transform.eulerAngles.y = 300 * sec;
    //  }
    //  if (tag.name == "mask-1.fbx") {
    //    transform.eulerAngles.y = 300 * sec;
    //  }
    //  if (tag.name == "fireplace.fbx") {
    //    transform.eulerAngles.x = std::sin(2 * sec);
    //    transform.eulerAngles.y = std::cos(3 * sec + 1.5f);
    //    transform.eulerAngles.z = std::cos(7 * sec + 0.3f);
    //  }
    //}
  }
  // find the loop maximum time
  float loop_time = 0.f;
  for (auto& [entity, transform, anim] : view) {
    for (auto& channel : anim.ani.channels) {
      auto& sampler = anim.ani.samplers[channel.samplerIndex];
      loop_time = std::max(loop_time, sampler.inputs.back());
    }
  }
  for (auto& [entity, transform, anim] : view) {
    for (auto& channel : anim.ani.channels) {
      auto& sampler = anim.ani.samplers[channel.samplerIndex];
      if (sampler.inputs.size() == 0) continue;
      float prev_time = 0.f;
      float sec_mod = std::min(std::fmod(sec, loop_time), sampler.inputs.back());
      for (int i = 0; i < sampler.inputs.size(); ++i) {
        float const curr_time = sampler.inputs[i];
        if (sec_mod >= prev_time && sec_mod <= curr_time) {
          switch (channel.path) {
          case GFX::AnimationComponent::AnimationChannel::PathType::TRANSLATION: {
            Math::vec3 val_curr = sampler.outputsVec3[i];
            Math::vec3 val_prev = (i == 0) ? val_curr:sampler.outputsVec3[i-1];
            Math::vec3 val;
            if (prev_time == curr_time) { val = val_curr; }
            else { float const alpha = (sec_mod - prev_time) / (curr_time - prev_time);
              val = Math::lerp(alpha, val_prev, val_curr); }
            transform.translation = val;
          } break;
          case GFX::AnimationComponent::AnimationChannel::PathType::ROTATION: {
              Math::vec4 val_curr = sampler.outputsVec4[i];
              Math::vec4 val_prev = (i == 0) ? val_curr:sampler.outputsVec4[i-1];
              Math::Quaternion quat_curr(val_curr.x, val_curr.y, val_curr.z, val_curr.w);
              Math::Quaternion quat_prev(val_prev.x, val_prev.y, val_prev.z, val_prev.w);
              Math::Quaternion quat;
              if (prev_time == curr_time) {
              quat = quat_curr;
              } else {
              float const alpha = (sec_mod - prev_time) / (curr_time - prev_time);
              quat = Math::slerp(alpha, quat_prev, quat_curr);
              }

              Math::mat3 quat_transform = quat.toMat3();
              Math::vec3 euler = Math::RotationMatrixToEulerAngles(quat_transform);
              euler.x *= 180. / Math::double_Pi;
              euler.y *= 180. / Math::double_Pi;
              euler.z *= 180. / Math::double_Pi;
              transform.eulerAngles = euler;
              transform.quaternion = quat;
              //transform.eulerAngles = euler;

              //Math::mat4 test2 = Math::mat4::rotateZ(euler.z) *
              //      Math::mat4::rotateY(euler.y) *
              //      Math::mat4::rotateX(euler.x);

              //double qx = quat.data[0];
              //double qy = quat.data[1];
              //double qz = quat.data[2];
              //double qw = quat.data[3];

              //double roll =
              //    atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
              //double pitch = -std::asin(2 * (qw * qy - qz * qx));
              //double yaw =
              //    atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
              //roll *= 180. / Math::double_Pi;
              //pitch *= 180. / Math::double_Pi;
              //yaw *= 180. / Math::double_Pi;
              //transform.eulerAngles = {static_cast<float>(-roll),
              //                         static_cast<float>(pitch),
              //                         static_cast<float>(yaw)};

                bool a = 1.f;
          } break;
          case GFX::AnimationComponent::AnimationChannel::PathType::SCALE: {
            Math::vec3 val_curr = sampler.outputsVec3[i];
            Math::vec3 val_prev = (i == 0) ? val_curr : sampler.outputsVec3[i - 1];
            Math::vec3 val;
            if (prev_time == curr_time) { val = val_curr; }
            else { float const alpha = (sec_mod - prev_time) / (curr_time - prev_time);
              val = Math::lerp(alpha, val_prev, val_curr); }
            transform.scale = val;
          } break;
          default: break;
          }
          break;
        }
        prev_time = curr_time;
      }
    }
  }
}

auto update_transforms(GFX::Scene& scene) noexcept -> void {
  std::stack<float> is_odd_scaling;
  std::stack<Math::mat4> global_transfrom;
  is_odd_scaling.push(1.f);
  global_transfrom.push(Math::mat4{});
  std::function<void(GFX::GameObject & go)> fn_process_per_node =
      [&](GFX::GameObject& go) {
        Math::mat4 objectMat = global_transfrom.top();
        float oddScaling = is_odd_scaling.top();
        GFX::TransformComponent* transform =
            go.getEntity().getComponent<GFX::TransformComponent>();
        objectMat = objectMat * transform->getTransform();
        //{
        //  Math::vec3 x = rot(transform * Vec3(1, 0, 0);
        //  Math::vec3 y = rotation * Vec3(0, 1, 0);
        //  Math::vec3 z = rotation * Vec3(0, 0, 1);

        //  // Next, scale the basis vectors
        //  x = x * transform->scale.x;
        //  y = y * transform->scale.y;
        //  z = z * transform->scale.z;
        //  // Extract the position of the transform
        //  Math::vec3 p = transform->translation;
        //  // Create matrix
        //  return Math::mat4(x.x, x.y, x.z, 0,  // X basis (& Scale)
        //              y.x, y.y, y.z, 0,  // Y basis (& scale)
        //              z.x, z.y, z.z, 0,  // Z basis (& scale)
        //              p.x, p.y, p.z, 1   // Position
        //  );


        //}
        oddScaling *= transform->scale.x * transform->scale.y * transform->scale.z;
        // update transform information
        transform->previousTransform = transform->transform;
        if (oddScaling != 0) oddScaling / std::abs(oddScaling);
        transform->oddScaling = oddScaling >= 0;
        transform->transform = objectMat;
        // process children nodes
        global_transfrom.push(objectMat);
        is_odd_scaling.push(oddScaling);
        for (auto child : go.children)
          fn_process_per_node(*scene.getGameObject(child));
        is_odd_scaling.pop();
        global_transfrom.pop();
      };
  // if the gameobject is a root in the scene, start processing transformations.
  for (auto go_handle : scene.gameObjects) {
    if (go_handle.second.parent == GFX::NULL_GO) {
      fn_process_per_node(go_handle.second);
    }
  }
}
}