#ifndef _APPART_PARTICLE_INCLUDE_
#define _APPART_PARTICLE_INCLUDE_

#extension GL_EXT_scalar_block_layout : require
layout(binding = 0, set = 0, scalar) buffer  _OldPosBuffer   { vec3 oldX[]; };
layout(binding = 1, set = 0, scalar) buffer  _LastPosBuffer  { vec3 lastX[]; };


#endif