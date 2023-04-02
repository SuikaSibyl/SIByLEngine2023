#version 460
layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(binding = 0, set = 0) uniform sampler2D texture_src;

vec3 ACESToneMapping(vec3 color, float adapted_lum) {
	const float A = 2.51f;
	const float B = 0.03f;
	const float C = 2.43f;
	const float D = 0.59f;
	const float E = 0.14f;

	color *= adapted_lum;
	return (color * (A * color + B)) / (color * (C * color + D) + E);
}

void main() {
    vec3 color = texture(texture_src, uv).rgb;
    // color = ACESToneMapping(color, 1.0f);

    // float gamma = 2.2;
    // color = pow(color, vec3(1.0/gamma));

    outColor = vec4(color, 1.0f);
}