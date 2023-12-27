#ifndef _SRENDERER_PMM_HEADER_
#define _SRENDERER_PMM_HEADER_

#include "cpp_compatible.hlsli"

interface IParametricModel {
};

struct PMM_Data {
    float3 point;
    float distance;
    float weight;
};


struct PMM<PM : IParametricModel, let N : int> {
    float responsibility(int component, float3 x) {
        return 0.f;
    }
    
    void stepwiseEM<let batch_size : int>(vector<PMM_Data, batch_size> dataset) {
        // E-step: update sufficient statistics
        // ---------------------------------------------------------------------
        vector<float, N> pdf;
        // Process the whole batch at once
        for (int i = 0; i < batch_size; ++i) {
            // compute posterior probabilities
            for(int j = 0; j < N; ++j) { pdf[j] = 0.f; }
            float denom = 0.f;
            const float dataWeight = dataset[i].weight;
            const float3 x = dataset[i].point;
            const float invDist = dataset[i].distance;

            for (int h = 0; h < N; ++h) {
                pdf[h] = responsibility(h, x);
                denom += pdf[h];
            }

        //     auto stats = model.getStats();
        //     Float eta = std::pow(stats.getCount() + 1.f, -cfg.alpha);
        //     /// update weight statistic (common for all components)
        //     stats.updateWeight(eta, dataWeight);

        //     for (int h = 0; h < model.nComponents(); ++h) {
        //         Float posterior = pdf[h] / denom;
        //         stats.updateComponent(h, eta, x, dataWeight, posterior, invDist);
        //     }
        //     stats.increment();
        }

        // M-step: update model
        // ---------------------------------------------------------------------

        // if (model.getStats().getCount() + 1 > cfg.delayedUpdate) {
        //     for (int h = 0; h < model.nComponents(); ++h) {
        //         model.update(h, cfg.regularizationType);
        //     }
        // }
    }
}

#endif // _SRENDERER_PMM_HEADER_