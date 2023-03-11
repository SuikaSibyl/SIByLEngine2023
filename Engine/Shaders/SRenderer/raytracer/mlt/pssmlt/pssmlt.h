#ifndef _SRENDERER_SBDPT_COMMON_HEADER_
#define _SRENDERER_SBDPT_COMMON_HEADER_

#include "../include/common_trace.h"
#include "../include/common_rgen.h"

struct PushConstants { 
    uvec2 resolution;
    uint sample_batch; 
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

struct PSSMLT_PrimarySample {
    float value = 0.;
    int64_t lastModificationIteration = 0;
    float valueBackup = 0;
    int64_t modifyBackup = 0;
};

struct PSSMLT_Sampler {
    uint rng;
    float sigmal;   // controls the size of “small step” mutations
    float largeStepProbability; // the probability of taking a “large step” mutation
    int  streamCount;
    int streamIndex, sampleIndex;
};

int getNextIndex(in PSSMLT_Sampler mlt_sampler) {
   return mlt_sampler.streamIndex + mlt_sampler.streamCount * mlt_sampler.sampleIndex++;
}

float get1D(in PSSMLT_Sampler mlt_sampler) {

}

/**
* For small steps, we apply normally distributed perturbations to each component.
* The advantage of sampling with a normal distribution like this is that it naturally tries a variety of mutation sizes.
* 
*/
float perturb(
    in const float value,
    in const float s1,
    in const float s2,
    inout uint RND
) {
	float result;
	float r = rnd();
	if (r < 0.5) {
		r = r * 2.0;
		result = value + s2 * exp(-log(s2 / s1) * r);
        if (result > 1.0) result -= 1.0;
	} else {
		r = (r - 0.5) * 2.0;
		result = value - s2 * exp(-log(s2 / s1) * r);
        if (result < 0.0) result += 1.0;
	}
	return result;
}

#endif