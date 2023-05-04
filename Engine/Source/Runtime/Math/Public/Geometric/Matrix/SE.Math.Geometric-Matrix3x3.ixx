module;
#include <cstdint>
export module SE.Math.Geometric:Matrix3x3;
import :Vector3;

namespace SIByL::Math {
export template <class T>
struct Matrix3x3 {
  Matrix3x3() = default;
  Matrix3x3(T const mat[3][3]);
  Matrix3x3(T t00, T t01, T t02, T t10, T t11, T t12, T t20, T t21, T t22);

  union {
    T data[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  };
};

template <class T>
Matrix3x3<T>::Matrix3x3(T const mat[3][3]) {
  memcpy(&(data[0][0]), &(mat[0][0]), sizeof(T) * 9);
}

template <class T>
Matrix3x3<T>::Matrix3x3(T t00, T t01, T t02, T t10, T t11, T t12, T t20, T t21,
                        T t22) {
  data[0][0] = t00;
  data[0][1] = t01;
  data[0][2] = t02;
  data[1][0] = t10;
  data[1][1] = t11;
  data[1][2] = t12;
  data[2][0] = t20;
  data[2][1] = t21;
  data[2][2] = t22;
}

export template <class T>
inline auto mul(Matrix3x3<T> const& m, Vector3<T> const& v) noexcept
    -> Vector3<T> {
  Vector3<T> s;
  for (size_t i = 0; i < 3; ++i) {
    s.data[i] = m.data[i][0] * v.data[0] + m.data[i][1] * v.data[1] +
                m.data[i][2] * v.data[2];
  }
  return s;
}

export template <class T>
auto operator*(Matrix3x3<T> const& m, Vector3<T> const& v) -> Vector3<T> {
  return mul<T>(m, v);
}

export using mat3 = Matrix3x3<float>;
export using imat3 = Matrix3x3<int32_t>;
export using umat3 = Matrix3x3<uint32_t>;
}  // namespace SIByL::Math