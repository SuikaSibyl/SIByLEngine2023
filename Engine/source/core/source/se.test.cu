#include "include.hpp"
#include <iostream>

__global__ void add2_kernel(float* c,
    const float* a,
    const float* b,
    int n) {
    int i = threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_add2(float* c,
    const float* a,
    const float* b,
    int n) {
    std::cout << "what" << std::endl;
    add2_kernel << <1, 32 >> > (c, a, b, n);
}