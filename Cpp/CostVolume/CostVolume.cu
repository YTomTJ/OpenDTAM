#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "CostVolume.cuh"

namespace cv { namespace cuda { namespace dtam_updateCost {

    cudaStream_t localStream;

#define CONSTT                                                                                     \
    uint rows, uint cols, uint layers, uint layerStep, float *hdata, float *cdata, float *lo,      \
        float *hi, float *loInd, float3 *base, float *bf, cudaTextureObject_t tex
#define CONSTS rows, cols, layers, layerStep, hdata, cdata, lo, hi, loInd, base, bf, tex

#define BLOCK_X 64
#define BLOCK_Y 4

    __global__ void globalWeightedBoundsCost(m34 p, float weight, CONSTT);
    void globalWeightedBoundsCostCaller(m34 p, float weight, CONSTT)
    {
        dim3 dimBlock(BLOCK_X, BLOCK_Y);
        dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
        globalWeightedBoundsCost<<<dimGrid, dimBlock, 0, localStream>>>(p, weight, CONSTS);
        assert(localStream);
        cudaSafeCall(cudaGetLastError());
    }

    __global__ void globalWeightedBoundsCost(m34 p, float weight, CONSTT)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        float xf = x;
        float yf = y;
        unsigned int offset = x + y * cols;
        float3 B = base[offset]; // Known bug:this requires 12 loads instead of 4 because
                                 // of stupid memory addressing, can't really fix
        // p.{u,v,0,1}
        float wi = p.data[8] * xf + p.data[9] * yf + p.data[11]; // far
        float xi = p.data[0] * xf + p.data[1] * yf + p.data[3]; // u
        float yi = p.data[4] * xf + p.data[5] * yf + p.data[7]; // v
        float minv = 1000.0, maxv = 0.0;
        float mini = 0;
        for (unsigned int l = 0; l < layers; l++) {
            float c0 = cdata[offset + l * layerStep];
            // {u,v,far} + {0,0,ds}*l = {u,v,d}
            float wiz = wi + p.data[10] * l;
            float xiz = xi + p.data[2] * l;
            float yiz = yi + p.data[6] * l;
            float4 c = tex2D<float4>(tex, xiz / wiz, yiz / wiz);
            float v1 = fabsf(c.x - B.x);
            float v2 = fabsf(c.y - B.y);
            float v3 = fabsf(c.z - B.z);
            float del = v1 + v2 + v3;
            float ns;
            del = .0001 * del + fminf(del, .01f) * 1.0f / .01f;
            ns = c0 * weight + (del) * (1 - weight);
            cdata[offset + l * layerStep] = ns;
            if (ns < minv) {
                minv = ns;
                mini = l;
            }
            maxv = fmaxf(ns, maxv);
        }
        lo[offset] = minv;
        loInd[offset] = mini;
        hi[offset] = maxv;
    }

}}} // namespace cv::cuda::dtam_updateCost
