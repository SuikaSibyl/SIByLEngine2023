module;
#include <cmath>
#include <cstdint>
export module Math.Transform:Transform;
import Math.Matrix;
import Math.Vector;
import Math.Geometry;
import Math.Trigonometric;

namespace SIByL::Math
{
	export struct Transform
	{
	public:
		Transform() = default;
		Transform(float const mat[4][4]);
		Transform(mat4 const& m);
		Transform(mat4 const& m, mat4 const& mInverse);
		Transform(Quaternion const& q);

		auto isIdentity() const noexcept -> bool;
		auto hasScale() const noexcept -> bool;
		auto swapsHandness() const noexcept -> bool;

		auto operator==(Transform const& t) const -> bool;
		auto operator!=(Transform const& t) const -> bool;

		auto operator*(point3 const& p) const->point3;
		auto operator*(vec3 const& v) const->vec3;
		auto operator*(normal3 const& n) const->normal3;
		auto operator*(ray3 const& s) const->ray3;
		auto operator*(bounds3 const& b) const->bounds3;
		auto operator*(Transform const& t2) const->Transform;

		auto operator()(point3 const& p, vec3& absError) const->point3;
		auto operator()(point3 const& p, vec3 const& pError, vec3& tError) const->point3;
		auto operator()(vec3 const& v, vec3& absError) const->vec3;
		auto operator()(vec3 const& v, vec3 const& pError, vec3& tError) const->vec3;
		auto operator()(ray3 const& r, vec3& oError, vec3& dError) const->ray3;

		friend auto inverse(Transform const& t) noexcept -> Transform;
		friend auto transpose(Transform const& t) noexcept -> Transform;

		mat4 m;
		mat4 mInv;
	};

	export inline auto inverse(Transform const& t) noexcept -> Transform;
	export inline auto transpose(Transform const& t) noexcept -> Transform;

	export inline auto translate(vec3 const& delta) noexcept -> Transform;
	export inline auto scale(float x, float y, float z) noexcept -> Transform;
	export inline auto rotateX(float theta) noexcept -> Transform;
	export inline auto rotateY(float theta) noexcept -> Transform;
	export inline auto rotateZ(float theta) noexcept -> Transform;
	export inline auto rotate(float theta, vec3 const& axis) noexcept -> Transform;

	export inline auto lookAt(point3 const& pos, point3 const& look, vec3 const& up) noexcept -> Transform;

	export inline auto orthographic(float zNear, float zFar) noexcept -> Transform;
	export inline auto perspective(float fov, float n, float f) noexcept -> Transform;
	export inline auto perspective(float fov, float aspect, float n, float f) noexcept -> Transform;

	/** Decompose an affine transformation into Translation x Rotation x Scaling */
	export inline auto decompose(mat4 const& m, vec3* t, Quaternion* rquat, mat4* s) noexcept -> void;
	export inline auto decompose(mat4 const& m, vec3* t, vec3* r, vec3* s) noexcept -> void;
}