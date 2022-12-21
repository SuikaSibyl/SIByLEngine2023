#ifndef _SRENDERER_COMMON_TRACE_HEADER_
#define _SRENDERER_COMMON_TRACE_HEADER_

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require

#include "common_descriptor_sets.h"

layout(binding = 0, set = 1) uniform accelerationStructureEXT tlas;
layout(binding = 1, set = 1, rgba32f) uniform image2D storageImage;

#endif