#version 460
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec3 color;
layout(location = 0) out vec4 outColor;

#ifdef __USE_DEPTH__

layout(binding = 0) uniform sampler2D depth;

void main() {
    const ivec2 pix = ivec2(gl_FragCoord.xy);
    const float depth = texelFetch(depth, pix, 0).x;
    float alpha = 1.f;
    if (depth < gl_FragDepth)
        alpha = 0.5f;
    outColor = vec4(color, alpha);
}

#else

void main() {
    outColor = vec4(color, 1.0);
}

#endif