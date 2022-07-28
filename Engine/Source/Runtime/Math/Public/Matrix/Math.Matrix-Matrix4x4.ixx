module;
#include <cstring>
#include <cstdint>
#include <memory>
#include <xmmintrin.h>
export module Math.Matrix:Matrix4x4;
import :Matrix3x3;
import Math.SIMD;

namespace SIByL::Math
{
	export template <class T>
		struct Matrix4x4
	{
		Matrix4x4() = default;
		Matrix4x4(T const mat[4][4]);
		Matrix4x4(T t00, T t01, T t02, T t03,
				  T t10, T t11, T t12, T t13,
				  T t20, T t21, T t22, T t23,
				  T t30, T t31, T t32, T t33);

		auto operator==(Matrix4x4<T> const& t) const -> bool;
		auto operator!=(Matrix4x4<T> const& t) const -> bool;
		auto operator-() const ->Matrix4x4<T>;
		operator Matrix3x3<T>() const;

		T data[4][4] = {
			{1,0,0,0},
			{0,1,0,0},
			{0,0,1,0},
			{0,0,0,1},
		};
	};

	template <class T>
	Matrix4x4<T>::Matrix4x4(T const mat[4][4])
	{
		memcpy(&(data[0][0]), &(mat[0][0]), sizeof(T) * 16);
	}

	template <class T>
	Matrix4x4<T>::Matrix4x4(T t00, T t01, T t02, T t03,
							T t10, T t11, T t12, T t13,
							T t20, T t21, T t22, T t23,
							T t30, T t31, T t32, T t33)
	{
		data[0][0] = t00; data[0][1] = t01; data[0][2] = t02; data[0][3] = t03;
		data[1][0] = t10; data[1][1] = t11; data[1][2] = t12; data[1][3] = t13;
		data[2][0] = t20; data[2][1] = t21; data[2][2] = t22; data[2][3] = t23;
		data[3][0] = t30; data[3][1] = t31; data[3][2] = t32; data[3][3] = t33;
	}

	template <class T>
	auto Matrix4x4<T>::operator==(Matrix4x4<T> const& t) const -> bool
	{
		return (memcmp(&(data[0][0]), &(t.data[0][0]), sizeof(T) * 16) == 0) ? true : false;
	}

	template <class T>
	auto Matrix4x4<T>::operator!=(Matrix4x4<T> const& t) const -> bool
	{
		return !(*this == t);
	}

	template <class T>
	auto Matrix4x4<T>::operator-() const -> Matrix4x4<T>
	{
		return Matrix4x4<T>{
			-data[0][0], -data[0][1], -data[0][2], -data[0][3], 
			-data[1][0], -data[1][1], -data[1][2], -data[1][3],
			-data[2][0], -data[2][1], -data[2][2], -data[2][3],
			-data[3][0], -data[3][1], -data[3][2], -data[3][3]
		};
	}

	template <class T>
	Matrix4x4<T>::operator Matrix3x3<T>() const
	{
		return Matrix3x3<T>{
			data[0][0], data[0][1], data[0][2], 
			data[1][0], data[1][1], data[1][2], 
			data[2][0], data[2][1], data[2][2], 
		};
	}

	export template <class T>
		inline auto transpose(Matrix4x4<T> const& m) noexcept -> Matrix4x4<T>
	{
		return Matrix4x4<T>(
			m.data[0][0], m.data[1][0], m.data[2][0], m.data[3][0],
			m.data[0][1], m.data[1][1], m.data[2][1], m.data[3][1],
			m.data[0][2], m.data[1][2], m.data[2][2], m.data[3][2],
			m.data[0][3], m.data[1][3], m.data[2][3], m.data[3][3]);
	}

	export template <class T>
		inline auto mul(Matrix4x4<T> const& m1, Matrix4x4<T> const& m2) noexcept -> Matrix4x4<T>
	{
		Matrix4x4<T> s;
		for (size_t i = 0; i < 4; ++i)
			for (size_t j = 0; j < 4; ++j)
				s.data[i][j] = m1.data[i][0] * m2.data[0][j] +
							   m1.data[i][1] * m2.data[1][j] +
							   m1.data[i][2] * m2.data[2][j] +
							   m1.data[i][3] * m2.data[3][j];
		return s;
	}

	export template <>
		inline auto mul(Matrix4x4<float> const& m1, Matrix4x4<float> const& m2) noexcept -> Matrix4x4<float>
	{
		Matrix4x4<float> s;
		
		__m128 v, result;
		__m128 const mrow0 = _mm_load_ps(&m2.data[0][0]);
		__m128 const mrow1 = _mm_load_ps(&m2.data[1][0]);
		__m128 const mrow2 = _mm_load_ps(&m2.data[2][0]);
		__m128 const mrow3 = _mm_load_ps(&m2.data[3][0]);

		for (int i = 0; i < 4; ++i) {
			v = _mm_load_ps(&m1.data[i][0]);
			result = _mm_mul_ps(_mm_replicate_x_ps(v), mrow0);
			result = _mm_add_ps(_mm_mul_ps(_mm_replicate_y_ps(v), mrow1), result);
			result = _mm_add_ps(_mm_mul_ps(_mm_replicate_z_ps(v), mrow2), result);
			result = _mm_add_ps(_mm_mul_ps(_mm_replicate_w_ps(v), mrow3), result);
			_mm_store_ps(&s.data[i][0], result);
		}
		return s;
	}

	export template <class T>
		inline auto inverse(Matrix4x4<T> const& m) noexcept -> Matrix4x4<T>
	{
		Matrix4x4<T> origin;
		Matrix4x4<T> inv;
		memcpy(&(origin.data[0][0]), &(m.data[0][0]), sizeof(T) * 16);
		
		for (int j = 0; j < 4; ++j) {
			for (int i = 0; i < 4; ++i) {
				if (i == j) continue;
				T p = origin.data[i][j];
				for (int k = 0; k < 4; ++k) {
					origin.data[i][k] = origin.data[j][j] * origin.data[i][k] - p * origin.data[j][k];
					inv.data[i][k] = origin.data[j][j] * inv.data[i][k] - p * inv.data[j][k];
				}
			}
		}

		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				inv.data[i][j] /= origin.data[i][i];

		return inv;
	}

	export using mat4 = Matrix4x4<float>;
	export using dmat4 = Matrix4x4<double>;
	export using imat4 = Matrix4x4<int32_t>;
	export using umat4 = Matrix4x4<uint32_t>;
}