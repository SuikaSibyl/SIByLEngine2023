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

const float k_sqrt_2 = 1.41421356237309504880;

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

float erfinv(in const float a) {
    // https://stackoverflow.com/a/49743348
    float p;
    float t = log(max(fma(a, -a, 1), 1.17549e-38));

    if (abs(t) > 6.125f) {          // maximum ulp error = 2.35793
        p = 3.03697567e-10f;             //  0x1.4deb44p-32
        p = fma(p, t, 2.93243101e-8f);   //  0x1.f7c9aep-26
        p = fma(p, t, 1.22150334e-6f);   //  0x1.47e512p-20
        p = fma(p, t, 2.84108955e-5f);   //  0x1.dca7dep-16
        p = fma(p, t, 3.93552968e-4f);   //  0x1.9cab92p-12
        p = fma(p, t, 3.02698812e-3f);   //  0x1.8cc0dep-9
        p = fma(p, t, 4.83185798e-3f);   //  0x1.3ca920p-8
        p = fma(p, t, -2.64646143e-1f);  // -0x1.0eff66p-2
        p = fma(p, t, 8.40016484e-1f);   //  0x1.ae16a4p-1
    } else {                             // maximum ulp error = 2.35456
        p = 5.43877832e-9f;              //  0x1.75c000p-28
        p = fma(p, t, 1.43286059e-7f);   //  0x1.33b458p-23
        p = fma(p, t, 1.22775396e-6f);   //  0x1.49929cp-20
        p = fma(p, t, 1.12962631e-7f);   //  0x1.e52bbap-24
        p = fma(p, t, -5.61531961e-5f);  // -0x1.d70c12p-15
        p = fma(p, t, -1.47697705e-4f);  // -0x1.35be9ap-13
        p = fma(p, t, 2.31468701e-3f);   //  0x1.2f6402p-9
        p = fma(p, t, 1.15392562e-2f);   //  0x1.7a1e4cp-7
        p = fma(p, t, -2.32015476e-1f);  // -0x1.db2aeep-3
        p = fma(p, t, 8.86226892e-1f);   //  0x1.c5bf88p-1
    }
    return a * p;

}


float minComponent(in const vec3 v) {
    return min(v.x, min(v.y, v.z));
}

float maxComponent(in const vec3 v) {
    return max(v.x, max(v.y, v.z));
}

int maxDimension(in const vec3 v) {
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : 
           ((v.y > v.z) ? 1 : 2);
}

vec3 permute(in const vec3 v, int x, int y, int z) {
    return vec3(v[x], v[y], v[z]);
}

#endif
