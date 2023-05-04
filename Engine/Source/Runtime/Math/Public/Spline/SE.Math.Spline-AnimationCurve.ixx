module;
#include <algorithm>
#include <vector>
export module SE.Math.Spline:AnimationCurve;

namespace SIByL::Math {
export enum struct WrapMode {
  CLAMP,
  REPEAT,
  PINGPOMG,
};

export struct KeyFrame {
  float time;
  float value;
  float inTangent;
  float outTangent;
};

export struct AnimationCurve {
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

auto AnimationCurve::evaluate(float time) noexcept -> float {
  if (time > keyFrames.back().time) {  // right warp
    switch (preWrapMode) {
      case WrapMode::CLAMP:
        return keyFrames.back().value;
        break;
      case WrapMode::REPEAT: {
        int passCount = int((time - keyFrames.back().time) /
                            (keyFrames.back().time - keyFrames.front().time));
        time = time - (passCount + 1) *
                          (keyFrames.back().time - keyFrames.front().time);
      } break;
      case WrapMode::PINGPOMG: {
        int passCount = int((time - keyFrames.back().time) /
                            (keyFrames.back().time - keyFrames.front().time));
        bool needReverse = (passCount % 2 == 0);
        time = time - (passCount + 1) *
                          (keyFrames.back().time - keyFrames.front().time);
        if (needReverse)
          time = keyFrames.front().time + keyFrames.back().time - time;
      } break;
      default:
        break;
    }
  } else if (time < keyFrames.front().time) {  // left warp
    switch (preWrapMode) {
      case WrapMode::CLAMP:
        return keyFrames.front().value;
        break;
      case WrapMode::REPEAT: {
        int passCount = int((keyFrames.front().time - time) /
                            (keyFrames.back().time - keyFrames.front().time));
        time = time + (passCount + 1) *
                          (keyFrames.back().time - keyFrames.front().time);
      } break;
      case WrapMode::PINGPOMG: {
        int passCount = int((keyFrames.front().time - time) /
                            (keyFrames.back().time - keyFrames.front().time));
        bool needReverse = (passCount % 2 == 0);
        time = time + (passCount + 1) *
                          (keyFrames.back().time - keyFrames.front().time);
        if (needReverse)
          time = keyFrames.front().time + keyFrames.back().time - time;
      } break;
      default:
        break;
    }
  }

  int left = 0;
  while (left + 1 < keyFrames.size()) {
    if (keyFrames[left].time <= time && keyFrames[left + 1].time > time) break;
    left++;
  }

  float t_l = 0;
  float t_r = 1;
  while (true) {
    float t = 0.5f * (t_l + t_r);
    Point point = evaluate(keyFrames[left], keyFrames[left + 1], t);
    float error = std::abs(point.time - time);
    if (error < errorTolerence)
      return point.value;
    else if (point.time < time)
      t_l = t;
    else
      t_r = t;
  }
}

auto AnimationCurve::evaluate(KeyFrame const& keyframe0,
                              KeyFrame const& keyframe1, float t) noexcept
    -> Point {
  // regular Cubic Hermite spline with tangents defined by hand
  float dt = keyframe1.time - keyframe0.time;
  float m0 = keyframe0.outTangent * dt;
  float m1 = keyframe1.inTangent * dt;

  float t2 = t * t;
  float t3 = t2 * t;

  float a = 2 * t3 - 3 * t2 + 1;
  float b = t3 - 2 * t2 + t;
  float c = t3 - t2;
  float d = -2 * t3 + 3 * t2;

  float time = a * keyframe0.time + b * m0 + c * m1 + d * keyframe1.time;
  float value = a * keyframe0.value + b * m0 + c * m1 + d * keyframe1.value;
  return Point{time, value};
}

auto compareKeyFrameByTime(KeyFrame const& lhv, KeyFrame const& rhv) noexcept
    -> bool {
  return lhv.time < rhv.time;
}

auto AnimationCurve::sortAllKeyFrames() noexcept -> void {
  std::sort(keyFrames.begin(), keyFrames.end(), compareKeyFrameByTime);
}
}  // namespace SIByL::Math