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

const static bool valgrind = 0;

// A test program to make the mapper run
using namespace cv;
using namespace cv::cuda;
using namespace std;

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

    cv::cuda::setDevice(0);
    isDeviceCompatible();
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    initGui();

    int ret = App_main(argc, argv);
    myExit();
    return ret;
}

void convertAhandaPovRayToStandard(const std::string &file, Mat &R, Mat &T)
{
    static const std::string delimiters(" =[]',;");

    cout << "text_file_name = " << file << endl;
    ifstream caminfo(file);
    if (!caminfo.is_open()) {
        cerr << "Failed to open param file, check location of sample trajectory!" << endl;
        exit(1);
    }

    Point3d direction;
    Point3d upvector;
    Point3d posvector;

    // cam_pos      = [112.831, 486.157, -264.376]';
    // cam_dir      = [0.44025, -0.317107, 0.840014]';
    // cam_up       = [0.0355246, 0.940977, 0.336602]';
    // cam_lookat   = [0, 0, 1]';
    // cam_sky      = [0, 1, 0]';
    // cam_right    = [1.19324, 0.157403, -0.565955]';
    // cam_fpoint   = [0, 0, 10]';
    // cam_angle    = 90;

    std::string line;
    while (std::getline(caminfo, line)) {

        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(delimiters), boost::token_compress_on);

        std::string field = parts[0];
        parts.erase(parts.begin());
        std::string cmd = boost::algorithm::join(parts, "\n");
        istringstream iss(cmd);

        if (field == "cam_dir") {
            iss >> direction.x;
            iss >> direction.z;
            iss >> direction.y;
        } else if (field == "cam_up") {
            iss >> upvector.x;
            iss >> upvector.z;
            iss >> upvector.y;
        } else if (field == "cam_pos") {
            iss >> posvector.x;
            iss >> posvector.z;
            iss >> posvector.y;
        }
    }

    R = Mat(3, 3, CV_64F);
    R.row(0) = Mat(direction.cross(upvector)).t();
    R.row(1) = Mat(-upvector).t();
    R.row(2) = Mat(direction).t();
    T = -R * Mat(posvector);
}

int App_main(int argc, char **argv)
{

#if !defined WIN32 && !defined _WIN32 && !defined WINCE && defined __linux__ && !defined ANDROID
    pthread_setname_np(pthread_self(), "App_main");
#endif

    const cv::Mat K0 = (Mat_<double>(3, 3) << 0.751875, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0);

    fs::path source(argv[1]);

    int numImg = 50;
    vector<Mat> images, Rs, Ts, Rs0, Ts0;

    std::vector<fs::path> image_files;
    for (auto const &e : fs::directory_iterator(source)) {
        if (fs::is_regular_file(e) && e.path().extension() == ".png")
            image_files.emplace_back(e.path());
    }

    {
        Mat image, R, T;
        for (int i = 0; i < numImg; ++i) {

            auto name = boost::str(boost::format("scene_%03d.png") % i);
            auto f = source / name;

            auto tf = f.parent_path() / (f.stem().string() + ".txt");
            convertAhandaPovRayToStandard(tf.string(), R, T);
            cout << "Opening: " << f.string() << endl;

            Mat image;
            imread(f.string(), -1).convertTo(image, CV_32FC3, 1.0 / 65535.0);
            if (image.rows != 480 || image.cols != 640) {
                cv::resize(image, image, Size(640, 480));
            }

            images.push_back(image.clone());
            Rs.push_back(R.clone());
            Ts.push_back(T.clone());
            Rs0.push_back(R.clone());
            Ts0.push_back(T.clone());
        }
    }

    HostMem cret(images[0].rows, images[0].cols, CV_32FC1);
    // Make a place to return downloaded images to
    Mat ret = cret.createMatHeader();

    // Setup camera matrix
    int layers = 32;
    int imagesPerCV = 20;

    cv::Mat K = K0.clone();
    K.at<double>(0, 0) *= images[0].cols;
    K.at<double>(1, 1) *= images[0].rows;
    K.at<double>(0, 2) *= images[0].cols - 1;
    K.at<double>(1, 2) *= images[0].rows - 1;
    CostVolume cv(images[0], (FrameID)0, layers, 0.015, 0.0, Rs[0], Ts[0], K);

    // Old Way
    int inc = 1;
    Mat image, R, T;
    cv::cuda::Stream s;

    for (int imageNum = 1; imageNum < numImg; imageNum++) {
        if (inc == -1 && imageNum < 4) {
            inc = 1;
        }
        T = Ts[imageNum].clone();
        R = Rs[imageNum].clone();
        image = images[imageNum];

        if (cv.count < imagesPerCV) { /// first it grabs enough frames to build a usable cost volume
            cv.updateCost(image, R, T); /// increments cv.count
            cudaDeviceSynchronize();

        } else {
            cudaDeviceSynchronize(); /// if there is a usable cost volume /// build regularized
                                     /// depthmap
            // Attach optimizer
            Ptr<DepthmapDenoiseWeightedHuber> dp
                = createDepthmapDenoiseWeightedHuber(cv.baseImageGray, cv.cvStream);
            DepthmapDenoiseWeightedHuber &denoiser = *dp;
            Optimizer optimizer(cv);
            optimizer.initOptimization();

            GpuMat a(cv.loInd.size(), cv.loInd.type());
            cv.loInd.copyTo(a, cv.cvStream);

            GpuMat d;
            denoiser.cacheGValues();
            ret = image * 0;

            cv.loInd.download(ret);
            pfShow("loInd", ret, 0, cv::Vec2d(0, layers));

            bool doneOptimizing;
            int Acount = 0;
            int QDcount = 0;
            do {
                a.download(ret);
                pfShow("A function", ret, 0, cv::Vec2d(0, layers));

                for (int i = 0; i < 10; i++) {
                    d = denoiser(a, optimizer.epsilon,
                        optimizer.getTheta()); // 10 iterations of denoiser, fed optimizer.epsilon
                    QDcount++;
                    d.download(ret);
                    pfShow("D function", ret, 0, cv::Vec2d(0, layers));
                }

                doneOptimizing = optimizer.optimizeA(d, a); // optimizeA(d,a)
                Acount++;
            } while (!doneOptimizing);

            // // optimizer.lambda=.05;
            // // optimizer.theta=10000;
            // // optimizer.optimizeA(a,a);
            optimizer.cvStream.waitForCompletion();

            // a.download(ret);
            // pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));

            Track tracker(cv); // tracking - find the pose transform to the current frame
            Mat out = optimizer.depthMap();
            double m;
            minMaxLoc(out, NULL, &m);
            tracker.depth = out * (.66 * cv.near / m);
            if (imageNum + imagesPerCV + 1 >= numImg) {
                inc = -1;
            }
            imageNum -= imagesPerCV + 1 - inc;
            for (int i = imageNum; i < numImg && i <= imageNum + imagesPerCV + 1; i++) {
                tracker.addFrame(images[i]);
                tracker.align();
                LieToRT(tracker.pose, R, T);
                Rs[i] = R.clone();
                Ts[i] = T.clone();

                Mat p, tp;
                p = tracker.pose;
                tp = RTToLie(Rs0[i], Ts0[i]);
                { // debug
                    cout << "True Pose: " << tp << endl;
                    cout << "True Delta: " << LieSub(tp, tracker.basePose) << endl;
                    cout << "Recovered Pose: " << p << endl;
                    cout << "Recovered Delta: " << LieSub(p, tracker.basePose) << endl;
                    cout << "Pose Error: " << p - tp << endl;
                }
                cout << i << endl;
                cout << Rs0[i] << Rs[i];
                // Display
                reprojectCloud(images[i], images[cv.fid], tracker.depth,
                    RTToP(Rs[cv.fid], Ts[cv.fid]), RTToP(Rs[i], Ts[i]), K);
            }
            cv = CostVolume(images[imageNum], (FrameID)imageNum, layers, 0.015, 0.0, Rs[imageNum],
                Ts[imageNum], K);
            s = optimizer.cvStream;
            // a.download(ret);
        }
        s.waitForCompletion(); // so we don't lock the whole system up forever
    }
    s.waitForCompletion();
    Stream::Null().waitForCompletion();
    return 0;
}
