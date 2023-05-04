module;
#include <cstdint>
export module SE.Math.Geometric:Matrix2x2;

namespace SIByL::Math {
export template <class T>
struct Matrix2x2 {
  union {
    T data[2][2];
  };
};

export using mat2 = Matrix2x2<float>;
export using imat2 = Matrix2x2<int32_t>;
export using umat2 = Matrix2x2<uint32_t>;
}  // namespace SIByL::Math