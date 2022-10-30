// Free for non-commercial, non-military, and non-critical
// use unless incorporated in OpenCV.
// Inherits OpenCV License if in OpenCV.

#include "CostVolume.hpp"
#include "CostVolume.cuh"

#include <opencv2/core/operations.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda/common.hpp>
// #include <opencv2/cudaimgproc.hpp>

#include "utils/utils.hpp"
#include "graphics.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::cuda;

void CostVolume::solveProjection(const cv::Mat &R, const cv::Mat &T)
{
    Mat P;
    RTToP(R, T, P);
    c.projection.create(4, 4, CV_64FC1);
    c.projection = 0.0;
    c.projection(Range(0, 2), Range(0, 3)) += _K.rowRange(0, 2);
    c.projection.at<double>(2, 3) = 1.0;
    c.projection.at<double>(3, 2) = 1.0;
    c.projection = c.projection * P;
    c.projection.at<double>(2, 2) = -_far;
    c.projection.row(2) /= _depthStep;

    // APPLICATION:
    // Backward:
    //  projection4x4^{-1}.{u,v,0,1} = P^{-1}.{(u-cx)/fx, (v-cy)/fy, 1, far} = .P^{-1}.X and then
    //  1/d * X = {x,y,z} projection4x4^{-1}.{0,0,1,0} = P^{-1}.{0,0,0,ds}
    // Forward: {x,y,z,1} = P.{x0,y0,z0,1}
    //  projection3x4.{x,y,z,1} / z = {(fxx+cxz_/z, (fyy+cyz)/z, (1/z-far)/ds, 1} =
    //  {u,v,(d-far)/ds,1} = {u,v,h,1} where h is the index of layer
    //      NOTE: u,v,h are in pixel indexing, i.e. 1...N / 0...(N-1)
    // Next:
    //  projection4x4 * projection4x4^{-1}.{u,v,0,1} = P'.P^{-1}.{u,v,far}
    //  projection4x4 * projection4x4^{-1}.{0,0,1,0} = P'.P^{-1}.{0,0,ds}
}

void CostVolume::checkInputs(const cv::Mat &R, const cv::Mat &T, const cv::Mat &_cameraMatrix)
{
    assert(R.size() == Size(3, 3));
    assert(R.type() == CV_64FC1);
    assert(T.size() == Size(1, 3));
    assert(T.type() == CV_64FC1);
    assert(_cameraMatrix.size() == Size(3, 3));
    assert(_cameraMatrix.type() == CV_64FC1);
    CV_Assert(_cameraMatrix.at<double>(2, 0) == 0.0);
    CV_Assert(_cameraMatrix.at<double>(2, 1) == 0.0);
    CV_Assert(_cameraMatrix.at<double>(2, 2) == 1.0);
}

#define FLATUP(src, dst)                                                                           \
    {                                                                                              \
        GpuMat tmp;                                                                                \
        tmp.upload(src);                                                                           \
        dst.create(1, rows *cols, src.type());                                                     \
        dst = dst.reshape(0, rows);                                                                \
    }
#define FLATALLOC(n)                                                                               \
    n.create(1, _rows *_cols, CV_32FC1);                                                           \
    n = n.reshape(0, _rows)

CostVolume::CostVolume(Mat image, FrameID _fid, int _layers, float _near, float _far, cv::Mat r,
    cv::Mat t, cv::Mat k, float initialCost, float initialWeight)
    : _initialWeight(initialWeight)
    , _rows(image.rows)
    , _cols(image.cols)
    , _layers(_layers)
    , _near(_near)
    , _far(_far)
    , _depthStep((_near - _far) / (_layers - 1))
    , _K(k.clone())
{
    c.fid = _fid;
    c.R = r;
    c.T = t;

    // For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
    CV_Assert(image.rows % 32 == 0 && image.cols % 32 == 0 && image.cols >= 64);

    checkInputs(c.R, c.T, _K);

    solveProjection(c.R, c.T); // stored as CostVolume::projection a 4x4 cv::Mat
    FLATALLOC(c.lo); // a cv::cuda::GpuMat
    FLATALLOC(c.hi);
    FLATALLOC(c.loInd);
    c.data_mat.create(_layers, _rows * _cols, CV_32FC1); // a cv::cuda::GpuMat

    Mat bwImage;
    image = image.reshape(0, 1);
    cv::cvtColor(image, bwImage, CV_RGB2GRAY);
    c.baseImage.upload(image); // a cv::cuda::GpuMat
    c.baseImageGray.upload(bwImage);
    c.baseImage = c.baseImage.reshape(0, _rows);
    c.baseImageGray = c.baseImageGray.reshape(0, _rows);

    c.loInd.setTo(Scalar(0, 0, 0), c.stream);
    c.data_mat.setTo(Scalar(initialCost), c.stream);

    c.data = (float *)c.data_mat.data;
    // hits = (float *)hitContainer.data;

    c.count = 0;

    // messy way to disguise cuda objects     // stored in CostVolume::  (private)
    _cuArray = Ptr<char>((char *)(new cudaArray_t));
    *((cudaArray **)(char *)_cuArray) = 0;
    _texObj = Ptr<char>((char *)(new cudaTextureObject_t));
    *((cudaTextureObject_t *)(char *)_texObj) = 0;
    ref = Ptr<char>(new char);
}

void CostVolume::simpleTex(const Mat &image, Stream stream)
{
    cudaArray_t &cuArray = *((cudaArray_t *)(char *)_cuArray);
    cudaTextureObject_t &texObj = *((cudaTextureObject_t *)(char *)_texObj);
    assert(image.isContinuous());
    assert(image.type() == CV_8UC4);

    // Describe texture
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    // {8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    // Fill Memory
    if (!cuArray) {
        cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, image.cols, image.rows));
    }

    assert((image.dataend - image.datastart) == image.cols * image.rows * sizeof(uchar4));

    cudaSafeCall(
        cudaMemcpyToArrayAsync(cuArray, 0, 0, image.datastart, image.dataend - image.datastart,
            cudaMemcpyHostToDevice, StreamAccessor::getStream(stream)));

    // Specify texture memory location
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    if (!texObj) {
        // Create texture object
        cudaSafeCall(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    }
    // return texObj;
}

void CostVolume::updateCost(const Mat &_image, const cv::Mat &R, const cv::Mat &T)
{
    using namespace cv::cuda::dtam_updateCost;
    localStream = cv::cuda::StreamAccessor::getStream(c.stream);

    // 0  1  2  3
    // 4  5  6  7
    // 8  9  10 11
    // 12 13 14 15
    //
    // 0 1 2
    // 3 4 5
    // 6 7 8
    //
    // want cudaReadModeNormalizedFloat for auto convert to [0,1]
    // cudaAddressModeClamp
    // cudaFilterModeLinear
    //
    // make sure we modify the K to take into account the texture coordinates
    //
    Mat image;
    {
        image = _image; // no copy
        if (_image.type() != CV_8UC4 || !_image.isContinuous()) {
            if (!_image.isContinuous() && _image.type() == CV_8UC4) {
                cBuffer.create(_image.rows, _image.cols, CV_8UC4);
                image = cBuffer; //.createMatHeader();
                _image.copyTo(image); // copies data
            }
            if (_image.type() != CV_8UC4) {
                cBuffer.create(_image.rows, _image.cols, CV_8UC4);
                Mat cm = cBuffer; //.createMatHeader();
                if (_image.type() == CV_8UC1 || _image.type() == CV_8SC1) {
                    cv::cvtColor(_image, cm, CV_GRAY2BGRA);
                } else if (_image.type() == CV_8UC3 || _image.type() == CV_8SC3) {
                    cv::cvtColor(_image, cm, CV_BGR2BGRA);
                } else {
                    image = _image;
                    if (_image.channels() == 1) {
                        cv::cvtColor(image, image, CV_GRAY2BGRA);
                    }
                    if (_image.channels() == 3) {
                        cv::cvtColor(image, image, CV_BGR2BGRA);
                    }
                    // image is now 4 channel, unknown depth but not 8 bit
                    if (_image.depth() >= 5) { // float
                        image.convertTo(cm, CV_8UC4, 255.0);
                    } else if (image.depth() >= 2) { // 0-65535
                        image.convertTo(cm, CV_8UC4, 1 / 256.0);
                    }
                }
                image = cm;
            }
        }
        CV_Assert(image.type() == CV_8UC4);
    }
    // change input image to a texture
    // ArrayTexture tex(image, stream);
    simpleTex(image, c.stream);
    cudaTextureObject_t &texObj = *((cudaTextureObject_t *)(char *)_texObj);
    //     cudaTextureObject_t texObj=simpleTex(image,stream);
    //     cudaSafeCall( cudaDeviceSynchronize() );

    // find projection matrix from cost volume to image (3x4)
    Mat viewMatrixImage;
    RTToP(R, T, viewMatrixImage);

    Mat camM(3, 4, CV_64FC1);
    camM = 0.0;
    _K.copyTo(camM(Range(0, 3), Range(0, 3)));

    // ytom: not exactly! <add 0.5 to x,y out //removing causes crash>
    // K(Range(0, 2), Range(2, 3)) += 0.5;

    Mat imFromWorld = camM * viewMatrixImage; // 3x4
    Mat imFromCV = imFromWorld * c.projection.inv();

    assert(c.baseImage.isContinuous());
    assert(c.lo.isContinuous());
    assert(c.hi.isContinuous());
    assert(c.loInd.isContinuous());

    double *p = (double *)imFromCV.data;
    m34 persp;
    for (int i = 0; i < 12; i++)
        persp.data[i] = p[i];

#define CONST_ARGS                                                                                 \
    _rows, _cols, _layers, _rows *_cols, NULL, c.data, (float *)(c.lo.data), (float *)(c.hi.data), \
        (float *)(c.loInd.data), (float3 *)(c.baseImage.data), (float *)c.baseImage.data, texObj

    float w = c.count++ + _initialWeight; // fun parse
    w /= (w + 1);
    assert(localStream);
    globalWeightedBoundsCostCaller(persp, w, CONST_ARGS); // calls Cuda Kernel
}

CostVolume &CostVolume::operator=(CostVolume &&rhs)
{
    _rows = rhs._rows;
    _cols = rhs._cols;
    _layers = rhs._layers;
    _near = rhs._near;
    _far = rhs._far;
    _depthStep = rhs._depthStep;
    _initialWeight = rhs._initialWeight;
    _K = std::move(rhs._K);
    this->c = std::move(rhs.c);
    this->_cuArray = std::move(rhs._cuArray);
    this->_texObj = std::move(rhs._texObj);
    this->cBuffer = std::move(rhs.cBuffer);
    this->ref = std::move(rhs.ref);
    return *this;
}

CostVolume::~CostVolume()
{
    cudaArray_t &cuArray = *((cudaArray_t *)(char *)_cuArray);
    cudaTextureObject_t &texObj = *((cudaTextureObject_t *)(char *)_texObj);
    // copy the Ptr without adding to refcount
    Ptr<char> *R = (Ptr<char> *)malloc(sizeof(Ptr<char>));
    memcpy(R, &ref, sizeof(Ptr<char>));
    int *rc = (((int **)(&ref))[1]);
    ref.release();
    cout << "destructor!: " << *rc << " arr: " << cuArray << endl;
    if (*rc <= 0) { // no one else has a copy of the cv, so we must clean up
        if (cuArray) {
            cudaFreeArray(cuArray);
            cuArray = 0;
        }
        if (texObj) {
            cudaDestroyTextureObject(texObj);
            texObj = 0;
        }
    }
    free(R);
}
