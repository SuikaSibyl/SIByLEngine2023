#version 460

layout(location = 0) out vec2 uv;

void main() {
    uint id = gl_VertexIndex;
	uv = vec2((id << 1) & 2, id & 2);
	gl_Position = vec4(uv * vec2(2) + vec2(-1), 0, 1);
}