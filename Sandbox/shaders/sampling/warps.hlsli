#ifndef _SRENDERER_SAMPLING_WARPS_HEADER_
#define _SRENDERER_SAMPLING_WARPS_HEADER_

// #include "common/"

interface IWarp1D {
    float forward(float u);
    float inverse(float u);
    float pdf(float u);
};

namespace iwarp {
struct warp2d_out {
    float2 sample;
    float pdf;
};
}

struct BilinearWarp {
    float w00;
    float w01;
    float w10;
    float w11;

    __init(float w00, float w01, float w10, float w11) {
        this.w00 = w00;
        this.w01 = w01;
        this.w10 = w10;
        this.w11 = w11;
    }

    // solve quadratic equation ax ^ 2 + bx + c = 0
    // return the positive solution if it exists, otherwise nan
    float solve_quadratic(float a, float b, float c) {
        float rad = b * b - 4 * a * c;
        // when rad < 0, no real solution
        if (rad < 0) return 0.f / 0.f;
        rad = sqrt(rad);
        // check both solutions and return the one in [0, 1]
        float tmp = (-b - rad) / (2 * a);
        // if tmp < 0 or tmp > 1:
        // tmp = (-b + rad) / (2 * a)
        tmp = (tmp < 0) ? (-b + rad) / (2 * a) : tmp;
        tmp = (tmp > 1) ? (-b + rad) / (2 * a) : tmp;
        tmp = (abs(a) < 1e-4) ? (-c / b) : tmp;
        return tmp;
    }

    float fu(float r) {
        float a = w10 + w11 - w01 - w00;
        float b = 2. * (w01 + w00);
        float c = -(w00 + w01 + w10 + w11) * r;
        return solve_quadratic(a, b, c);
    }

    float fv(float r, float u) {
        float a = (1. - u) * w01 + u * w11 - (1. - u) * w00 - u * w10;
        float b = 2. * (1 - u) * w00 + 2. * u * w10;
        float c = -((1. - u) * w01 + u * w11 +
                    (1. - u) * w00 + u * w10) * r;
        return solve_quadratic(a, b, c);
    }

    iwarp::warp2d_out inverse(float2 uv) {
        iwarp::warp2d_out o;
        const float u = fu(uv.x);
        const float v = fv(uv.y, u);
        o.sample = float2(u, v);
        o.pdf = 4. * (
            (1 - u) * (1 - v) * w00 +
            u * (1 - v) * w10 +
            (1 - u) * v * w01 +
            u * v * w11
        ) / (w00 + w01 + w10 + w11);
        return o;
    }

    float pdf(float2 uv) {
        const float u = uv.x;
        const float v = uv.y;
        return 4. * (
            (1 - u) * (1 - v) * w00 +
            u * (1 - v) * w10 +
            (1 - u) * v * w01 +
            u * v * w11
        ) / (w00 + w01 + w10 + w11);
    }
};

#endif // _SRENDERER_SAMPLING_WARPS_HEADER_