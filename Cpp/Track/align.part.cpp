/*
 *  Track_align.cpp
 *
 *
 *  Created by Paul Foster on 6/4/14.
 *
 *
 */

#include "Track.hpp"
#include "Align_part.cpp"
#include "tictoc.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
// needs:
// Mat base
// Mat cameraMatrix
// Mat depth
// Cost cv
// cols

// Models Used:
//
//  Warp is function of p, the parameters
//  Model 1: Template(:)=Image(Warp(:)+dWarp(:))
//  Model 2: Template(dWarp_inv(:))=Image(Warp(:))
//
//  nb: the "+" means composition, i.e. Warp(:)+dWarp(:)=dWarp(Warp)
//
//
//  J1=dI/dWarp*dWarp/dp=grad(I)(Warp)*dWarp/dp
//  J*dp=T-I
//  (J'J)*dp=J1'I
//  dp=(J'J)^-1*J'I     //A O(n*N) operation if J cacheable, else O(n^2*N) operation
//
//  The first model is more correct, since Image is smooth
//  but Template is not smooth.
//  However, the second allows caching the Jacobians and should
//  work if the depth is mostly smooth.
//  Both models are averaged for ESM, giving a second order method
//  but at the cost of being neither correct nor cachable.
//  However, ESM is good for initial alignment because no depth is
//  used, so the presumed map is smooth both ways. Might go faster
//  to just cache anyway though and do more non-ESM steps in same
//  amount of time.
//
//  The paper is clear that it uses ESM for the initial levels of
//  the pyramid, and implies that it uses Model 1 for the full
//  estimation. TODO:I would like to allow either choice to be made.
//
using namespace cv;
using namespace std;

#define LEVELS_2D 2

void createPyramid(const Mat &image, vector<Mat> &pyramid, int &levels)
{

    Mat in = image;
    if (levels == 0) { // auto size to end at >=15px tall (use height because shortest dim usually)
        for (float scale = 1.0; scale >= 15.0 / image.rows; scale /= 2, levels++)
            ;
    }
    assert(levels > 0);
    int l2 = levels - 1;
    pyramid.resize(levels);
    pyramid[l2--] = in;

    for (float scale = 0.5; l2 >= 0; scale /= 2, l2--) {
        Mat out;

        resize(in, out, Size(), .5, .5, CV_INTER_AREA);
        pyramid[l2] = out;
        in = out;
    }
}

static void createPyramids(const Mat &base, const Mat &depth, const Mat &input,
    const Mat &cameraMatrixIn, vector<Mat> &basePyr, vector<Mat> &depthPyr, vector<Mat> &inPyr,
    vector<Mat> &cameraMatrixPyr, int &levels)
{
    createPyramid(base, basePyr, levels);
    createPyramid(depth, depthPyr, levels);
    createPyramid(input, inPyr, levels);
    int l2 = 0;
    cameraMatrixPyr.resize(levels);
    // Figure out camera matrices for each level
    for (double scale = 1.0, l2 = levels - 1; l2 >= 0; scale /= 2, l2--) {
        Mat cameraMatrix = make4x4(cameraMatrixIn.clone());
        cameraMatrix(Range(0, 2), Range(2, 3)) += .5;
        cameraMatrix(Range(0, 2), Range(0, 3)) *= scale;
        cameraMatrix(Range(0, 2), Range(2, 3)) -= .5;
        cameraMatrixPyr[l2] = cameraMatrix;
    }
}

void Track::align() { align_gray(baseImage, depth, thisFrame); };

void Track::align_gray(Mat &_base, Mat &depth, Mat &_input)
{
    Mat input, base, lastFrameGray;
    input = makeGray(_input);
    base = makeGray(_base);
    lastFrameGray = makeGray(lastFrame);

    tic();
    int levels = 6; // 6 levels on a 640x480 image is 20x15
    int startlevel = 0;
    int endlevel = 6;

    Mat p = LieSub(pose, basePose); // the Lie parameters
    cout << "pose: " << p << endl;

    vector<Mat> basePyr, depthPyr, inPyr, cameraMatrixPyr;
    createPyramids(
        base, depth, input, cameraMatrix, basePyr, depthPyr, inPyr, cameraMatrixPyr, levels);

    vector<Mat> lfPyr;
    createPyramid(lastFrameGray, lfPyr, levels);

    int level = startlevel;
    Mat p2d = Mat::zeros(1, 6, CV_64FC1);
    for (; level < LEVELS_2D; level++) {
        int iters = 1;
        for (int i = 0; i < iters; i++) {
            // HACK: use 3d alignment with depth disabled for 2D. ESM would be much better, but I'm
            // lazy right now.
            align_level_largedef_gray_forward(
                lfPyr[level], // Total Mem cost ~185 load/stores of image
                depthPyr[level] * 0.0, inPyr[level],
                cameraMatrixPyr[level], // Mat_<double>
                p2d, // Mat_<double>
                CV_DTAM_FWD, 1, 3);
            //             if(tocq()>.01)
            //                 break;
        }
    }
    p = LieAdd(p2d, p);
    //     cout<<"3D iteration:"<<endl;
    for (level = startlevel; level < levels && level < endlevel; level++) {
        int iters = 1;
        for (int i = 0; i < iters; i++) {
            float thr
                = (levels - level) >= 2 ? .05 : .2; // more stringent matching on last two levels
            bool improved;
            improved = align_level_largedef_gray_forward(
                basePyr[level], // Total Mem cost ~185 load/stores of image
                depthPyr[level], inPyr[level],
                cameraMatrixPyr[level], // Mat_<double>
                p, // Mat_<double>
                CV_DTAM_FWD, thr, 6);

            //             if(tocq()>.5){
            //                 cout<<"completed up to level: "<<level-startlevel+1<<"   iter:
            //                 "<<i+1<<endl; goto loopend;//olny sactioned use of goto, the double
            //                 break
            //             }
            //             if(!improved){
            //                 break;
            //             }
        }
    }
loopend:

    pose = LieAdd(p, basePose);
    static int runs = 0;
    // assert(runs++<2);
    toc();
}
