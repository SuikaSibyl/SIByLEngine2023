#version 330
#ifdef GL_ARB_shading_language_420pack
#extension GL_ARB_shading_language_420pack : require
#endif

layout(binding = 1) uniform sampler2D in_color_0;

in vec2 _S2;
layout(location = 0) out vec4 _S1;

void main()
{
    _S1 = vec4(textureLod(in_color_0, _S2, 0.0).xyz, 1.0);
}

