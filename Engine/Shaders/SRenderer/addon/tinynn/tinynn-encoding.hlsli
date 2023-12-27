#ifndef _SRENDERER_ADDON_HALF_TINYNN_ENCODING_HLSLI_HEADER_
#define _SRENDERER_ADDON_HALF_TINYNN_ENCODING_HLSLI_HEADER_

#include "tinynn-tensorview.hlsli"

interface IPositionalEncoding<let C:int> {
    HalfFeature<C> encode_impl(HalfFeature<C> input);
};

struct FrequencyEncoding<let C : int> : IPositionalEncoding<C> {

    HalfFeature<C> encode_impl(HalfFeature<C> input) {
        const uint32_t encoded_input_feature_i = j / (n_frequencies * 2);
        const uint32_t log2_frequency = (j / 2) % n_frequencies;
        const float phase_shift = (j % 2) * (PI / 2);
        const float x = scalbnf(data_in(encoded_input_feature_i, i), log2_frequency);
        const float input = x * PI + phase_shift;
        data_out(j, i) = (T)__sinf(input);
    }

    
};

#endif // _SRENDERER_ADDON_HALF_TINYNN_ENCODING_HLSLI_HEADER_