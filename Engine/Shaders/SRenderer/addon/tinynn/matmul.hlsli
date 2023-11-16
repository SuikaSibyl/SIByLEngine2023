#ifndef _SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_
#define _SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_

void __inline_matmul_impl_32_16_16(uint input_ptr, uint weights_ptr, uint output_ptr) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "matmul.glsli")");
    __intrinsic_asm "wmma_inline_matmul_glsl_32_16_16($0, $1, $2)";
}
void __inline_matmul_impl_16_32_16(uint input_ptr, uint weights_ptr, uint output_ptr) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "matmul.glsli")");
    __intrinsic_asm "wmma_inline_matmul_glsl_16_32_16($0, $1, $2)";
}

void inline_set_input_element(int i, float16_t value) {
    __intrinsic_asm "set_input_buffer_glsl($0, $1)";
}
void inline_set_weights_element(int i, float16_t value) {
    __intrinsic_asm "set_weights_buffer_glsl($0, $1)";
}
float16_t inline_get_input_element(int i) {
    __intrinsic_asm "set_input_buffer_glsl($0)";
}
float16_t inline_get_weights_element(int i) {
    __intrinsic_asm "set_weights_buffer_glsl($0)";
}
float inline_get_output_element(int i) {
    __intrinsic_asm "get_output_buffer_glsl($0)";
}

#endif // !_SRENDERER_ADDON_TINYNN_MATMUL_HLSLI_HEADER_