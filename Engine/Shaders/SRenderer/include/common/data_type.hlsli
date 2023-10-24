#ifndef _SRENDERER_COMMON_DATA_TYPE_HEADER_
#define _SRENDERER_COMMON_DATA_TYPE_HEADER_

#ifndef DATA_TYPE_ENUM
#define DATA_TYPE_ENUM 0
#endif // !DATA_TYPE_ENUM

#ifdef DATA_TYPE_ENUM
// indicate a DATA_TYPE macro,
// 0: float, 1: float2, 2: float3, 3: float4
#if DATA_TYPE_ENUM == 0
#define DATA_TYPE float
#define DATA_COMPONENT 1
#elif DATA_TYPE_ENUM == 1
#define DATA_TYPE float2
#define DATA_COMPONENT 2
#elif DATA_TYPE_ENUM == 2
#define DATA_TYPE float3
#define DATA_COMPONENT 3
#elif DATA_TYPE_ENUM == 3
#define DATA_TYPE float4
#define DATA_COMPONENT 4
#endif // DATA_TYPE
#endif // DATA_TYPE_ENUM

#endif // !_SRENDERER_COMMON_DATA_TYPE_HEADER_