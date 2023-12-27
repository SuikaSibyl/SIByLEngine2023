#ifndef _SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_
#define _SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_

void __inline_set_half_shared_buffer(int i, float16_t value) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "half-matmul-include.glsli")");
    __intrinsic_asm "glsl_set_half_shared_buffer($0, $1)";
}

float16_t __inline_get_half_shared_buffer(int i) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "half-matmul-include.glsli")");
    __intrinsic_asm "glsl_get_half_shared_buffer($0)";
}

void __inline_wmma_128_16_16() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_128_16_16()";
}

void __inline_wmma_16_128_16() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_16_128_16()";
}

void __inline_wmma_128_32_32() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_128_32_32()";
}

void __inline_wmma_32_128_32() {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "half-matmul-include.glsli")");
    __intrinsic_asm "glsl_wmma_32_128_32()";
}

#endif // !_SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_