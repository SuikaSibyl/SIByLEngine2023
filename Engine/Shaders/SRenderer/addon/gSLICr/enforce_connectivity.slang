
[[vk::push_constant]]
cbuffer PushConstants {
    int2 img_size;
};

Texture2D<int> u_inIdxImg;
RWTexture2D<int> u_outIdxImg;

/**
 * We eliminate stray pixels with two one thread per pixel calls of the same kernel.
 * This prompts a pixel to change its label of that of the surrounding pixels (in a 2 × 2 neighborhood)
 * if all have a different label.
 */
void supress_local_lable(
    Texture2D<int> in_idx_img,
    RWTexture2D<int> out_idx_img,
    int2 img_size, int x, int y
) {
    const int2 pixel = int2(x, y);
    const int clable = in_idx_img[pixel];
    // don't suppress boundary
    if (x <= 1 || y <= 1 || x >= img_size.x - 2 || y >= img_size.y - 2) {
        out_idx_img[pixel] = clable;
        return;
    }

    int diff_count = 0;
    int diff_lable = -1;
    
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++) {
            // TODO: use shared memory
            int nlable = in_idx_img[int2(x + i, y + j)];
            if (nlable != clable) {
                diff_lable = nlable;
                diff_count++;
            }
        }
    
    out_idx_img[pixel] = (diff_count >= 16) ? diff_lable : clable;
}

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    if (any(DTid.xy >= img_size)) return;
    supress_local_lable(u_inIdxImg, u_outIdxImg, img_size, DTid.x, DTid.y);
}