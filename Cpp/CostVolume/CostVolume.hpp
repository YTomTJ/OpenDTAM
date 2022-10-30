// Free for non-commercial, non-military, and non-critical
// use unless incorporated in OpenCV.
// Inherits OpenCV Licence if in OpenCV.

#ifndef COSTVOLUME_HPP
#define COSTVOLUME_HPP

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

typedef int FrameID;

class CostVolume {
public:
    struct _context {
        FrameID fid;

        cv::Mat R;
        cv::Mat T;
        cv::Mat projection; // projects world coordinates (x,y,z) into (rows,cols,layers)

        cv::cuda::GpuMat baseImage;
        cv::cuda::GpuMat baseImageGray;
        cv::cuda::GpuMat lo; // min difference at reprojection
        cv::cuda::GpuMat hi; // max difference at reprojection
        cv::cuda::GpuMat loInd; // layer index of the min difference

        float *data;
        cv::cuda::GpuMat data_mat;
        // float *hits;
        // cv::cuda::GpuMat hitContainer;

        int count;
        cv::cuda::Stream stream;
    } c;

public:
    inline int rows() const { return _rows; };
    inline int cols() const { return _cols; };
    inline int layers() const { return _layers; };
    // inverse depth of center of voxels in layer layers-1
    inline float near() const { return _near; };
    // inverse depth of center of voxels in layer 0
    inline float far() const { return _far; };
    inline float depthStep() const { return _depthStep; };
    inline float initialWeight() const { return _initialWeight; };
    // Note! should be in OpenCV format
    const cv::Mat K() const { return _K; };

public:
    void updateCost(const cv::Mat &image, const cv::Mat &R,
        const cv::Mat &T); // Accepts pinned RGBA8888 or BGRA8888 for high speed

    CostVolume() = delete;
    CostVolume(CostVolume &&) = default;
    CostVolume(const CostVolume &) = delete;
    CostVolume(cv::Mat image, FrameID _fid, int _layers, float _near, float _far, cv::Mat R,
        cv::Mat T, cv::Mat _cameraMatrix, float initialCost = 3.0, float initialWeight = .001);
    ~CostVolume();

    cv::Mat depth() const;
    static cv::Mat depth(const cv::cuda::GpuMat &layers, float ds, float far);

    CostVolume &operator=(CostVolume &&);

    // // HACK: remove this function in release
    // cv::Mat downloadOldStyle(int layer)
    // {
    //     cv::Mat cost;
    //     cv::cuda::GpuMat tmp = dataContainer.rowRange(layer, layer + 1);
    //     tmp.download(cost);
    //     cost = cost.reshape(0, rows);
    //     return cost;
    // }

private:
    void solveProjection(const cv::Mat &R, const cv::Mat &T);
    void checkInputs(const cv::Mat &R, const cv::Mat &T, const cv::Mat &_cameraMatrix);
    void simpleTex(const cv::Mat &image, cv::cuda::Stream cvStream = cv::cuda::Stream::Null());

protected:
    int _rows;
    int _cols;
    int _layers;
    float _near; // inverse depth of center of voxels in layer layers-1
    float _far; // inverse depth of center of voxels in layer 0
    float _depthStep;
    float _initialWeight;
    cv::Mat _K; // Note! should be in OpenCV format

private:
    // temp variables ("static" containers)
    cv::Ptr<char> _cuArray; // Ptr<cudaArray*> really
    cv::Ptr<char> _texObj; // Ptr<cudaTextureObject_t> really
    cv::Mat cBuffer; // Must be pagable
    cv::Ptr<char> ref;
};

#endif // COSTVOLUME_HPP
