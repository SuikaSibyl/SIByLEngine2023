module;
#include <cstring>
#include <cstdint>
#include <memory>
#include <cmath>
#include <pmmintrin.h>
#include <xmmintrin.h>
export module SE.Math.Geometric:Matrix4x4;
import :Vector2;
import :Vector3;
import :Vector4;
import :Matrix3x3;
import SE.Math.Misc;

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

		static inline auto translate(Vector3<T> const& delta) noexcept -> Matrix4x4<T>;
		static inline auto scale(float x, float y, float z) noexcept -> Matrix4x4<T>;
		static inline auto scale(Vector3<T> const& scale) noexcept -> Matrix4x4<T>;
		static inline auto rotateX(float theta) noexcept -> Matrix4x4<T>;
		static inline auto rotateY(float theta) noexcept -> Matrix4x4<T>;
		static inline auto rotateZ(float theta) noexcept -> Matrix4x4<T>;
		static inline auto rotate(float theta, vec3 const& axis) noexcept -> Matrix4x4<T>;

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

	export template <class T>
	auto operator*(Matrix4x4<T> const& m1, Matrix4x4<T> const& m2) -> Matrix4x4<T> {
		return mul<T>(m1, m1);
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

	export template <>
	auto operator*(Matrix4x4<float> const& m1, Matrix4x4<float> const& m2) -> Matrix4x4<float> {
		return mul<float>(m1, m2);
	}

	export template <class T>
	inline auto mul(Matrix4x4<T> const& m, Vector4<T> const& v) noexcept -> Vector4<T>
	{
		Vector4<T> s;
		for (size_t i = 0; i < 4; ++i) {
			s.data[i] = m.data[i][0] * v.data[0]
					  + m.data[i][1] * v.data[1]
					  + m.data[i][2] * v.data[2]
					  + m.data[i][3] * v.data[3];
		}
		return s;
	}

	export template <class T>
	auto operator*(Matrix4x4<T> const& m, Vector4<T> const& v) -> Vector4<T> {
		return mul<T>(m, v);
	}

	export template <>
	inline auto mul(Matrix4x4<float> const& m, Vector4<float> const& v) noexcept -> Vector4<float>
	{
		Vector4<float> s;
		__m128 mrow0, mrow1, mrow2, mrow3;
		__m128 acc_0, acc_1, acc_2, acc_3;
		__m128 const vcol = _mm_load_ps(&v.data[0]);

		mrow0 = _mm_load_ps(&(m.data[0][0]));
		mrow1 = _mm_load_ps(&(m.data[1][0]));
		mrow2 = _mm_load_ps(&(m.data[2][0]));
		mrow3 = _mm_load_ps(&(m.data[3][0]));

		acc_0 = _mm_mul_ps(mrow0, vcol);
		acc_1 = _mm_mul_ps(mrow1, vcol);
		acc_2 = _mm_mul_ps(mrow2, vcol);
		acc_3 = _mm_mul_ps(mrow3, vcol);

		acc_0 = _mm_hadd_ps(acc_0, acc_1);
		acc_2 = _mm_hadd_ps(acc_2, acc_3);
		acc_0 = _mm_hadd_ps(acc_0, acc_2);
		_mm_store_ps(&s.data[0], acc_0);
		return s;
	}

	export template <>
	auto operator*(Matrix4x4<float> const& m, Vector4<float> const& v) -> Vector4<float> {
		return mul<float>(m, v);
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

	template <class T>
	auto Matrix4x4<T>::translate(Vector3<T> const& delta) noexcept -> Matrix4x4<T> {
		return Matrix4x4<T>(1, 0, 0, delta.x,
			0, 1, 0, delta.y,
			0, 0, 1, delta.z,
			0, 0, 0, 1);
	}

	template <class T>
	auto Matrix4x4<T>::scale(float x, float y, float z) noexcept -> Matrix4x4<T> {
		return Matrix4x4<T>(x, 0, 0, 0,
			0, y, 0, 0,
			0, 0, z, 0,
			0, 0, 0, 1);
	}
	
	template <class T>
	auto Matrix4x4<T>::scale(Vector3<T> const& scale) noexcept -> Matrix4x4<T> {
		return Matrix4x4<T>(scale.x, 0, 0, 0,
			0, scale.y, 0, 0,
			0, 0, scale.z, 0,
			0, 0, 0, 1);
	}

	template <class T>
	auto Matrix4x4<T>::rotateX(float theta) noexcept -> Matrix4x4<T> {
		float sinTheta = std::sin(Math::radians(theta));
		float cosTheta = std::cos(Math::radians(theta));
		return Matrix4x4<T>(1, 0, 0, 0,
			0, cosTheta, -sinTheta, 0,
			0, sinTheta, cosTheta, 0,
			0, 0, 0, 1);
	}

	template <class T>
	auto Matrix4x4<T>::rotateY(float theta) noexcept -> Matrix4x4<T> {
		float sinTheta = std::sin(Math::radians(theta));
		float cosTheta = std::cos(Math::radians(theta));
		Matrix4x4<T> m(T(cosTheta), 0, T(sinTheta), 0,
			0, 1, 0, 0,
			T(-sinTheta), 0, T(cosTheta), 0,
			0, 0, 0, 1);
		return m;
	}

	template <class T>
	auto Matrix4x4<T>::rotateZ(float theta) noexcept -> Matrix4x4<T> {
		float sinTheta = std::sin(Math::radians(theta));
		float cosTheta = std::cos(Math::radians(theta));
		Matrix4x4<T> m(cosTheta, -sinTheta, 0, 0,
			sinTheta, cosTheta, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		return m;
	}

	template <class T>
	auto Matrix4x4<T>::rotate(float theta, vec3 const& axis) noexcept -> Matrix4x4<T> {
		vec3 a = normalize(axis);
		float sinTheta = std::sin(Math::radians(theta));
		float cosTheta = std::cos(Math::radians(theta));
		Matrix4x4<T> m;
		// Compute rotation of first basis vector
		m.data[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
		m.data[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
		m.data[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
		m.data[0][3] = 0;
		// Compute rotations of second and third basis vectors
		m.data[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
		m.data[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
		m.data[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
		m.data[1][3] = 0;

		m.data[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
		m.data[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
		m.data[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
		m.data[2][3] = 0;
		return m;
	}

	export using mat4 = Matrix4x4<float>;
	export using dmat4 = Matrix4x4<double>;
	export using imat4 = Matrix4x4<int32_t>;
	export using umat4 = Matrix4x4<uint32_t>;
}