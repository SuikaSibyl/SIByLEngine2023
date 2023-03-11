#ifndef _MATH_HEADER_
#define _MATH_HEADER_

const float k_inf = 1.0 / 0.0;
const float k_nan = 0.0 / 0.0;

/** float value of PI */
const float k_pi = 3.14159265;
const float k_pi_times_2 = k_pi * 2;

const float k_pi_over_2 = k_pi / 2;
const float k_pi_over_4 = k_pi / 4;

const float k_inv_pi = 1. / k_pi;
const float k_inv_2_pi = 1. / (2*k_pi);

float atan2(in float y, in float x) {
    return x == 0.0 ? sign(y)*k_pi/2 : atan(y, x);
}

float length_square(in vec3 v) {
    return dot(v, v);
}

float distance_squared(in vec3 v0, in vec3 v1) {
    return dot(v0-v1, v0-v1);
}

float pow5(in float x) {
    float x2 = x * x;
    return x2 * x2 * x;
};

float abs_dot(in const vec3 v0, in const vec3 v1) {
    return abs(dot(v0, v1));
}

/**
* Numerically stable quadratic equation solver at^2 + bt + c = 0
* @return:  false when it can't find solutions.
* @ref:     https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
*/
bool quadratic(
    in float a, in float b, in float c, 
    out float t0, out float t1)
{
    // Degenerated case
    if (a == 0) {
        if (b == 0)
            return false;
        t0 = t1 = -c / b;
        return true;
    }
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
        return false;
    float root_discriminant = sqrt(discriminant);
    if (b >= 0) {
        t0 = (- b - root_discriminant) / (2 * a);
        t1 = 2 * c / (- b - root_discriminant);
    } else {
        t0 = 2 * c / (- b + root_discriminant);
        t1 = (- b + root_discriminant) / (2 * a);
    }
    return true;
}

#endif
