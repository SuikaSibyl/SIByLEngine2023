#ifndef _SRENDERER_COMMON_DEFINITION_CAMERA_HEADER_
#define _SRENDERER_COMMON_DEFINITION_CAMERA_HEADER_

/**
* Universal camera data for both rasterizer & raytracer.
* Definition & annotations are modified from Falcor:
* @ref: https://github.com/NVIDIAGameWorks/Falcor
*/
struct CameraData {
    mat4     viewMat;                   ///< Camera view matrix.
    mat4     prevViewMat;               ///< Camera view matrix associated to previous frame.
    mat4     projMat;                   ///< Camera projection matrix.
    mat4     viewProjMat;               ///< Camera view-projection matrix.
    mat4     invViewProj;               ///< Camera inverse view-projection matrix.
    mat4     viewProjMatNoJitter;       ///< Camera view-projection matrix. No jittering is applied!
    mat4     prevViewProjMatNoJitter;   ///< Camera view-projection matrix associated to previous frame. No jittering is applied!
    mat4     projMatNoJitter;           ///< Camera projection matrix. No jittering is applied!

    vec3     posW;                      ///< Camera world-space position.
    float    focalLength;               ///< Camera focal length in mm. Default is 59 degree vertical, 90 horizontal FOV at 16:9 aspect ratio.
    vec3     prevPosW;                  ///< Camera world-space position associated to previous frame.
    float    rectArea;                  ///< Recrtangles area A at z=1 plane.
    vec3     up;                        ///< Camera world-space up vector.
    float    aspectRatio;               ///< Camera film frame aspect ratio, equal to frameWidth / frameHeight
    vec3     target;                    ///< Camera target point in world-space.
    float    nearZ;                     ///< Camera near plane.
    vec3     cameraU;                   ///< Camera base vector U. Normalized it indicates the right image plane vector. The length is dependent on the FOV.
    float    farZ;                      ///< Camera far plane.
    vec3     cameraV;                   ///< Camera base vector V. Normalized it indicates the up image plane vector. The length is dependent on the FOV.
    float    jitterX;                   ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    vec3     cameraW;                   ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float    jitterY;                   ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    float    frameHeight;               ///< Camera film frame height in mm. 24 is the height of a 35mm film
    float    frameWidth;                ///< Camera film frame width in mm.  42 2/3 is the width assuming 24mm height and a 16:9 aspect ratio
    float    focalDistance;             ///< Camera focal distance in scene units.
    float    apertureRadius;            ///< Camera aperture radius in scene units.
    float    shutterSpeed;              ///< Camera shutter speed in seconds.
    float    ISOSpeed;                  ///< Camera film speed based on ISO standards.
    float   _padding1;
    float   _padding2;
};

#endif