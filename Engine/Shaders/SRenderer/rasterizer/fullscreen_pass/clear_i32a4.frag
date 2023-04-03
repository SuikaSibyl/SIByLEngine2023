#version 460
layout(location = 0) in vec2 uv;

layout(location = 0) out float r_channel;
layout(location = 1) out float g_channel;
layout(location = 2) out float b_channel;
layout(location = 3) out float a_channel;

void main() {
    r_channel = 0;
    g_channel = 0;
    b_channel = 0;
    a_channel = 0;
}