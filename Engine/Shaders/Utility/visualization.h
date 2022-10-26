#ifndef _VISUALIZATION_HEADER_
#define _VISUALIZATION_HEADER_

/**
* Visualization Module. 
*/

/** Map a float value between [0,1] to a RGB color. */
vec3 colorMap_naive(in float val) {
  const float fraction = 2 * clamp(val, 0, 1) - 1;
  if (fraction < -0.5f)         return vec3(0.0f, 2*(fraction+1.0f), 1.0f);
  else if (fraction < 0.0f)     return vec3(0.0f, 1.0f, 1.0f - 2.0f * (fraction + 0.5f));
  else if (fraction < 0.5f)     return vec3(2.0f * fraction, 1.0f, 0.0f);
  else return vec3(1.0f, 1.0f - 2.0f*(fraction - 0.5f), 0.0f);
}


#endif