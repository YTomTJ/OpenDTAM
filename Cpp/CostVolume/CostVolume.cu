#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "CostVolume.cuh"

namespace cv { namespace cuda { namespace dtam_updateCost {

    cudaStream_t localStream;

    __global__ void globalWeightedBoundsCost(m34 proj, PARAMS)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        float u = x;
        float v = y;
        unsigned int offset = x + y * cols;
        float3 B = base[offset]; // Known bug:this requires 12 loads instead of 4 because
                                 // of stupid memory addressing, can't really fix
        // p . {u,v,h,0}
        float xx = proj.data[0] * u + proj.data[1] * v + /* p.data[2]  * h */ +proj.data[3]; // x
        float yy = proj.data[4] * u + proj.data[5] * v + /* p.data[6]  * h */ +proj.data[7]; // y
        float dd = proj.data[8] * u + proj.data[9] * v + /* p.data[10] * h */ +proj.data[11]; // z
        float minv = 1000.0, maxv = 0.0, mini = 0;
        // Try for each inverse depth layers.
        for (unsigned int h = 0; h < layers; h++) {
            float _xi = xx + proj.data[2] * h;
            float _yi = yy + proj.data[6] * h;
            float _di = dd + proj.data[10] * h;
            // cost data
            float c0 = cdata[offset + h * layerStep];
            float4 c = tex2D<float4>(tex, _xi / _di, _yi / _di);
            float v1 = fabsf(c.x - B.x);
            float v2 = fabsf(c.y - B.y);
            float v3 = fabsf(c.z - B.z);
            float del = v1 + v2 + v3;
            float ns;
            del = .0001 * del + fminf(del, .01f) * 1.0f / .01f;
            ns = c0 * weight + (del) * (1 - weight);
            cdata[offset + h * layerStep] = ns;
            if (ns < minv) {
                minv = ns;
                mini = h;
            }
            maxv = fmaxf(ns, maxv);
        }
        lo[offset] = minv;
        loInd[offset] = mini;
        hi[offset] = maxv;
    }

#define BLOCK_X 64
#define BLOCK_Y 4
#define CONSTS  weight, rows, cols, layers, layerStep, NULL, cdata, lo, hi, loInd, base, bf, tex

    void globalWeightedBoundsCostCaller(m34 p, PARAMS)
    {
        dim3 dimBlock(BLOCK_X, BLOCK_Y);
        dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
        globalWeightedBoundsCost<<<dimGrid, dimBlock, 0, localStream>>>(p, CONSTS);
        assert(localStream);
        cudaSafeCall(cudaGetLastError());
    }

}}} // namespace cv::cuda::dtam_updateCost
