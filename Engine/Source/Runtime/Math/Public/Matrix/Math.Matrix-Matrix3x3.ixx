module;
#include <cstdint>
export module Math.Matrix:Matrix3x3;

namespace SIByL::Math
{
	export template <class T>
		struct Matrix3x3
	{
		Matrix3x3() = default;
		Matrix3x3(T const mat[3][3]);
		Matrix3x3(T t00, T t01, T t02,
				  T t10, T t11, T t12,
				  T t20, T t21, T t22);

		union
		{
			T data[3][3] = {
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1}
			};
		};
	};

	template <class T>
	Matrix3x3<T>::Matrix3x3(T const mat[3][3])
	{
		memcpy(&(data[0][0]), &(mat[0][0]), sizeof(T) * 9);
	}

	template <class T>
	Matrix3x3<T>::Matrix3x3(
		T t00, T t01, T t02,
		T t10, T t11, T t12,
		T t20, T t21, T t22)
	{
		data[0][0] = t00; data[0][1] = t01; data[0][2] = t02;
		data[1][0] = t10; data[1][1] = t11; data[1][2] = t12;
		data[2][0] = t20; data[2][1] = t21; data[2][2] = t22;
	}

	export using mat3 = Matrix3x3<float>;
	export using imat3 = Matrix3x3<int32_t>;
	export using umat3 = Matrix3x3<uint32_t>;
}