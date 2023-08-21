
struct svoxel_info {
    vec3 color_info;    // avg color of the superpixel
    int no_voxels;      // number of voxels
    vec3 center;        // center (avg pixel pos) of the supervoxel
    int id;             // superpixel id
};

ivec3 ReconstructIndex(int flattenID, int dimension) {
    return ivec3(flattenID % dimension, (flattenID / dimension) % dimension, flattenID / (dimension * dimension));
}