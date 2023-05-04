cmake_minimum_required(VERSION 3.23)
set(PROJECT_NAME Math)
set(CMAKE_CXX_STANDARD 20)

################################################################################
# Source groups
################################################################################
set(MOD_LIST
        # SE.Math.Misc Module
        Public/Misc/SE.Math.Misc-RoundingError.ixx
        Public/Misc/SE.Math.Misc-EquationSolving.ixx
        Public/Misc/SE.Math.Misc-RNG.ixx
        Public/Misc/SE.Math.Misc-Common.ixx
        Public/Misc/SE.Math.Misc-Limits.ixx
        Public/Misc/SE.Math.Misc-SIMD.ixx
        Public/Misc/SE.Math.Misc-Trigonometric.ixx
        Public/Misc/SE.Math.Misc.ixx
        # SE.Math.FixedPoint Module
        #        Public/Deterministic/SE.Math.FixedPoint.ixx;
        # SE.Math.Geometric Module
        Public/Geometric/Vector/SE.Math.Geometric-Vector2.ixx
        Public/Geometric/Vector/SE.Math.Geometric-Vector3.ixx
        Public/Geometric/Vector/SE.Math.Geometric-Vector4.ixx
        Public/Geometric/Matrix/SE.Math.Geometric-Matrix2x2.ixx
        Public/Geometric/Matrix/SE.Math.Geometric-Matrix3x3.ixx
        Public/Geometric/Matrix/SE.Math.Geometric-Matrix4x4.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Point2.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Point3.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Normal3.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Ray3.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Bounds2.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Bounds3.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Geometric.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Integral.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Quaternions.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-Transform.ixx
        Public/Geometric/Geometry/SE.Math.Geometric-AnimatedTransform.ixx
        Public/Geometric/SE.Math.Geometric.ixx
        # SE.Math.Spline Module
        Public/Spline/SE.Math.Spline-AnimationCurve.ixx;
        Public/Spline/SE.Math.Spline.ixx;
        )
set(SRC_LIST
        )

set(ALL_FILES
        ${MOD_LIST}
        ${SRC_LIST}
        )

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
    cmake_minimum_required(VERSION 3.26)
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API 2182bf5c-ef0d-489a-91da-49dbc3090d2a)
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")

    add_library(${PROJECT_NAME})
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${SRC_LIST}
            PRIVATE FILE_SET CXX_MODULES FILES
            ${MOD_LIST}
            )

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    add_library(${PROJECT_NAME})
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${SRC_LIST}
            ${MOD_LIST}
            )

endif ()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Core)