// Essential-matrix estimation in normalized camera coordinates.
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <cstring>

extern "C" int pmtt_pose(int n, const double* a, const double* b,
                         double threshold, int seed, double* Eout,
                         double* Rout, double* tout, unsigned char* maskout) {
    try {
        cv::setNumThreads(1);
        cv::setRNGSeed(seed);
        cv::Mat p1(n, 2, CV_64F, const_cast<double*>(a));
        cv::Mat p2(n, 2, CV_64F, const_cast<double*>(b));
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F), mask;
        cv::Mat E = cv::findEssentialMat(p1, p2, K, cv::USAC_MAGSAC,
                                       .999, threshold, 10000, mask);
        if (E.empty() || E.rows != 3 || E.cols != 3) return -1;
        cv::Mat R, t, triangulated;
        int good = cv::recoverPose(E, p1, p2, K, R, t, 1e9, mask, triangulated);
        if (good < 5) return -2;
        std::memcpy(Eout, E.ptr<double>(), 9 * sizeof(double));
        std::memcpy(Rout, R.ptr<double>(), 9 * sizeof(double));
        std::memcpy(tout, t.ptr<double>(), 3 * sizeof(double));
        std::memcpy(maskout, mask.ptr<unsigned char>(), n);
        return good;
    } catch (const cv::Exception&) {
        return -3;
    } catch (...) {
        return -4;
    }
}
