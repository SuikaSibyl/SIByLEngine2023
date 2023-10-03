/**
 *
 */

#ifndef _SRENDERER_PATHGUIDING_PDF_ACCUMULATOR_
#define _SRENDERER_PATHGUIDING_PDF_ACCUMULATOR_

RWTexture2D<uint> u_PdfAccumulator;
RWTexture2D<uint> u_PdfAccumulatorInfo;

void AddOneSample2Info() {
    InterlockedAdd(u_PdfAccumulatorInfo[int2(0)], 1u);
}

void AddSomeSamples2Info(uint count) {
    InterlockedAdd(u_PdfAccumulatorInfo[int2(0)], count);
}

void AddOneSample2Map(
    in int2 pixel,
    in int2 resolution
) {
    if (any(pixel < 0 || pixel >= resolution)) return;
    InterlockedAdd(u_PdfAccumulator[pixel], 1);
}

#endif // _SRENDERER_PATHGUIDING_PDF_ACCUMULATOR_