#ifndef SRENDERER_COMMON_MIPMAP_HEADER
#define SRENDERER_COMMON_MIPMAP_HEADER

/** Compute the number of mipmaps levels for a given texture size.
 * @param width The width of the texture.
 * @param height The height of the texture.
 * @return The number of (full-chained) mipmaps levels.
 */
int compute_mipmap_levels(int width, int height) {
    return 1 + int(floor(log2(max(width, height))));
}

/**
 * Compute the size of a mipmap level.
 * @param size The size of the original texture.
 * @param level The level of the mipmap.
 * @return The size of the mipmap.
 */
int2 mipmap_size(int2 size, int level) {
    return max(1, size >> level);
}

#endif // !SRENDERER_COMMON_MIPMAP_HEADER