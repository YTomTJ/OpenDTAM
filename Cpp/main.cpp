#include <iostream>
#include <opencv2/core/core.hpp>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

// Mine
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/CostVolume.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
#include "Optimizer/Optimizer.hpp"
#include "graphics.hpp"
#include "Track/Track.hpp"
#include "utils/utils.hpp"

// debug
#include "tictoc.h"

int App_main(int argc, char **argv);

void myExit() { ImplThread::stopAllThreads(); }

bool isDeviceCompatible()
{
    int device_id = cv::cuda::getDevice();
    if (device_id < 0)
        return false;

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);

    if (cv::cuda::TargetArchs::hasEqualOrLessPtx(major, minor))
        return true;

    for (int i = minor; i >= 0; i--)
        if (cv::cuda::TargetArchs::hasBin(major, i))
            return true;

    return false;
}

int main(int argc, char **argv)
{
    if (argc < 2)
        throw std::runtime_error("Path of data is not given.");
    if (argc < 3)
        throw std::runtime_error("Path of camera is not given.");

    cv::cuda::setDevice(0);
    isDeviceCompatible();
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    initGui();

    int ret = App_main(argc, argv);
    myExit();
    return ret;
}

#include <yaml-cpp/yaml.h>

cv::Mat load_intrinsics(const std::string &file)
{
    auto node = YAML::LoadFile(file);
    cv::Mat intrinsics(3, 3, CV_64FC1);
    for (int i = 0; i < node["intrinsics"].size() && i < 3; ++i) {
        auto in = node["intrinsics"][i].as<std::vector<double>>();
        intrinsics.at<double>(i, 0) = in[0];
        intrinsics.at<double>(i, 1) = in[1];
        intrinsics.at<double>(i, 2) = in[2];
    }
    return intrinsics;
}

void load_pose(const std::string &file, cv::Mat &R, cv::Mat &T)
{
    try {
        std::ifstream ifs(file);
        R = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0.0));
        T = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0.0));
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                ifs >> R.at<double>(r, c);
            }
            ifs >> T.at<double>(r, 0);
        }
    } catch (std::exception &e) {
        throw std::runtime_error(std::string("Load pose from ") + file + " failed." + e.what());
    }
}

int App_main(int argc, char **argv)
{
#if !defined WIN32 && !defined _WIN32 && !defined WINCE && defined __linux__ && !defined ANDROID
    pthread_setname_np(pthread_self(), "App_main");
#endif

    using namespace cv;
    using namespace cv::cuda;

    const double z_near = 1.0; // mm
    const double z_far = 250.0; // mm
    // const double near = 1.0 / z_near; // mm^{-1}
    // const double far = 1.0 / z_far; // mm^{-1}
    const double near = 0.01;
    const double far = 0.0;
    const int layers = 32;
    const int imagesPerCV = 20;
    const int numImg = 50;
    const int W = 640;
    const int H = 480;

    fs::path camera(argv[2]);
    const Mat K0 = load_intrinsics(camera.string());

    fs::path source(argv[1]);
    std::vector<Mat> images, Rs, Ts, Rs0, Ts0;

    std::vector<fs::path> image_files;
    for (auto const &e : fs::directory_iterator(source)) {
        if (fs::is_regular_file(e) && e.path().extension() == ".png")
            image_files.emplace_back(e.path());
    }

    // Sort by alphabetically
    std::sort(image_files.begin(), image_files.end(), [](auto &a, auto &b) { return a < b; });

    {
        Mat image, R, T;
        for (int i = 0; i < numImg; ++i) {

            auto f = image_files[i];
            std::cout << "Opening: " << f.string() << std::endl;

            auto tf = f.parent_path() / (f.stem().string() + ".txt");
            load_pose(tf.string(), R, T);
            Rs.push_back(R.clone());
            Ts.push_back(T.clone());
            Rs0.push_back(R.clone());
            Ts0.push_back(T.clone());

            auto img = imread(f.string(), -1);
            img.convertTo(image, CV_32FC3, 1.0 / 65535.0);
            if (image.rows != H || image.cols != W) {
                resize(image, image, Size(W, H));
            }
            images.push_back(image.clone());
        }
    }

    // Setup camera matrix
    Mat K = K0.clone();
    K.at<double>(0, 0) *= images[0].cols;
    K.at<double>(1, 1) *= images[0].rows;
    K.at<double>(0, 2) *= images[0].cols - 1;
    K.at<double>(1, 2) *= images[0].rows - 1;
    CostVolume cv(images[0], (FrameID)0, layers, near, far, Rs[0], Ts[0], K);

    cuda::HostMem cret(images[0].rows, images[0].cols, CV_32FC1);
    // Make a place to return downloaded images to
    Mat ret = cret.createMatHeader();

    // Old Way
    int inc = 1;
    Mat image, R, T;
    cuda::Stream s;

    for (int imageNum = 1; imageNum < numImg; imageNum++) {
        if (inc == -1 && imageNum < 4) {
            inc = 1;
        }
        T = Ts[imageNum].clone();
        R = Rs[imageNum].clone();
        image = images[imageNum];

        if (cv.c.count
            < imagesPerCV) { /// first it grabs enough frames to build a usable cost volume
            cv.updateCost(image, R, T); /// increments cv.c.count
            cudaDeviceSynchronize();

        } else {
            cudaDeviceSynchronize(); /// if there is a usable cost volume /// build regularized
                                     /// depthmap
            // Attach optimizer
            Ptr<DepthmapDenoiseWeightedHuber> dp
                = createDepthmapDenoiseWeightedHuber(cv.c.baseImageGray, cv.c.stream);
            DepthmapDenoiseWeightedHuber &denoiser = *dp;
            Optimizer optimizer(cv);
            optimizer.initOptimization();

            GpuMat a(cv.c.loInd.size(), cv.c.loInd.type());
            cv.c.loInd.copyTo(a, cv.c.stream);

            GpuMat d;
            denoiser.cacheGValues();
            ret = image * 0;

            cv.c.loInd.download(ret);
            pfShow("loInd", ret, 0, Vec2d(0, layers));

            cv.c.baseImage.download(ret);
            pfShow("Source", ret, 0);

            bool doneOptimizing;
            int Acount = 0;
            int QDcount = 0;
            do {
                pfShow("A function", CostVolume::depth(a, cv.depthStep(), cv.far()), 0,
                    Vec2d(1.0 / near, 5000.0));

                for (int i = 0; i < 10; i++) {
                    d = denoiser(a, optimizer.epsilon,
                        optimizer.getTheta()); // 10 iterations of denoiser, fed optimizer.epsilon
                    QDcount++;
                    d.download(ret);
                    pfShow("D function", ret, 0, Vec2d(0, layers));
                }

                doneOptimizing = optimizer.optimizeA(d, a); // optimizeA(d,a)
                Acount++;
            } while (!doneOptimizing);

            // // optimizer.lambda=.05;
            // // optimizer.theta=10000;
            // // optimizer.optimizeA(a,a);
            optimizer.cvStream.waitForCompletion();

            // // a.download(ret);
            // // pfShow("A function loose", ret, 0, Vec2d(0, layers));

            // Track tracker(cv); // tracking - find the pose transform to the current frame
            // Mat out = optimizer.depthMap();
            // double m;
            // minMaxLoc(out, NULL, &m);
            // tracker.depth = out * (.66 * cv.near() / m);
            // if (imageNum + imagesPerCV + 1 >= numImg) {
            //     inc = -1;
            // }
            // imageNum -= imagesPerCV + 1 - inc;
            // for (int i = imageNum; i < numImg && i <= imageNum + imagesPerCV + 1; i++) {
            //     tracker.addFrame(images[i]);
            //     tracker.align();
            //     LieToRT(tracker.pose, R, T);
            //     Rs[i] = R.clone();
            //     Ts[i] = T.clone();

            //     Mat p, tp;
            //     p = tracker.pose;
            //     tp = RTToLie(Rs0[i], Ts0[i]);
            //     { // debug
            //         std::cout << "True Pose: " << tp << std::endl;
            //         std::cout << "True Delta: " << LieSub(tp, tracker.basePose) << std::endl;
            //         std::cout << "Recovered Pose: " << p << std::endl;
            //         std::cout << "Recovered Delta: " << LieSub(p, tracker.basePose) << std::endl;
            //         std::cout << "Pose Error: " << p - tp << std::endl;
            //     }
            //     std::cout << i << std::endl;
            //     std::cout << Rs0[i] << Rs[i];
            //     // Display
            //     reprojectCloud(images[i], images[cv.c.fid], tracker.depth,
            //         RTToP(Rs[cv.c.fid], Ts[cv.c.fid]), RTToP(Rs[i], Ts[i]), K);
            // }
            // cv = CostVolume(images[imageNum], (FrameID)imageNum, layers, near, far, Rs[imageNum],
            //     Ts[imageNum], K);
            s = optimizer.cvStream;
            // // a.download(ret);
        }
        s.waitForCompletion(); // so we don't lock the whole system up forever
    }
    s.waitForCompletion();
    Stream::Null().waitForCompletion();
    return 0;
}
