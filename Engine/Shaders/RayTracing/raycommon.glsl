struct hitPayload {
    vec3 hitValue;
};

// Returns the color of the sky in a given direction (in linear color space)
vec3 skyColor(vec3 direction) {
    // +y in world space is up, so:
    if(direction.y > 0.0f)
        return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
    else
        return vec3(0.03f);
}
