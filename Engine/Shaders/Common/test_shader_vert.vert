#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConsts {
  mat4 model;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragColor;

void main() {
    vec4 modelPos = model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * modelPos;
    fragColor = vec3(inUV, 0);
}