
/**
* @ref: https://www.shadertoy.com/view/MlcGDB
* @param P: point of current uv
* @param A: begin of the segment
* @param B: end of the segment
* @param r: radius of the segment
*/
float draw_segment(vec2 P, vec2 A, vec2 B, float r) {
    if(A==B) return 0.0;
    vec2 g = B - A;
    vec2 h = P - A;
    float d = length(h - g * clamp(dot(g, h) / dot(g,g), 0.0, 1.0));
	return smoothstep(r, 0.5*r, d);
}
