#ifndef _SRENDERER_COMMON_RESTIR_LIT_INCLUDED_
#define _SRENDERER_COMMON_RESTIR_LIT_INCLUDED_

// We could define custom sample lit and define _CUSTOM_WRS_SAMPLE_
// The sample structure is unrealted to the restir algorithm
#ifndef _CUSTOM_WRS_SAMPLE_
struct SampleLit {
    vec2 uv;
    uint index;
};
#endif

struct ReservoirLit {
    SampleLit y;     // the selected sample so far
    uint  M;         // the number of samples seen so far
    float wSum;      // the sum of weights
    float targetPDF; // p_hat of y
};

ReservoirLit EmptyReservoir() {
    ReservoirLit reservoir;
    reservoir.M = 0;
    reservoir.wSum = 0.0f;
    reservoir.targetPDF = 0.0f;
    return reservoir;
}

void UpdateReservoir (
    inout ReservoirLit reservoir,
    in const SampleLit x,
    float targetPDF, float sourcePDF,
    float rnd
) {
    const float w = targetPDF / sourcePDF;
    reservoir.wSum += w;
    reservoir.M += 1;
    if(rnd < (w / reservoir.wSum)) {
        reservoir.y = x;
        reservoir.targetPDF = targetPDF;
    }
}

float GetRISWeight(in const ReservoirLit reservoir) {
    return reservoir.wSum / (reservoir.M * reservoir.targetPDF); 
}

#endif