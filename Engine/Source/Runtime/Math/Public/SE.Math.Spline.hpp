#pragma once
#include <algorithm>
#include <vector>
#include <common_config.hpp>

namespace SIByL::Math {
SE_EXPORT enum struct WrapMode {
  CLAMP,
  REPEAT,
  PINGPOMG,
};

SE_EXPORT struct KeyFrame {
  float time;
  float value;
  float inTangent;
  float outTangent;
};

SE_EXPORT struct AnimationCurve {
 public:
  AnimationCurve() = default;
  AnimationCurve(std::initializer_list<KeyFrame> const& initializer_list)
      : keyFrames(initializer_list) {
    sortAllKeyFrames();
  }
  auto evaluate(float time) noexcept -> float;

  WrapMode preWrapMode = WrapMode::CLAMP;
  WrapMode postWrapMode = WrapMode::CLAMP;

 private:
  float errorTolerence = 0.00001f;
  std::vector<KeyFrame> keyFrames;
  struct Point {
    float time;
    float value;
  };
  auto evaluate(KeyFrame const& keyframe0, KeyFrame const& keyframe1,
                float t) noexcept -> Point;
  auto sortAllKeyFrames() noexcept -> void;
};
}  // namespace SIByL::Math