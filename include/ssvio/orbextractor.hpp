//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_ORBEXTRACTOR_HPP
#define SSVIO_ORBEXTRACTOR_HPP
#include "opencv2/opencv.hpp"
#include "memory"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include "glog/logging.h"
#include "ssvio/orbpattern.hpp"

namespace ssvio {

class ExtractorNode
{
 public:
  ExtractorNode(): bNoMore(false) { }

  void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3,
                  ExtractorNode &n4);

  std::vector<cv::KeyPoint> vKeys;
  cv::Point2i UL, UR, BL, BR;
  std::list<ExtractorNode>::iterator lit;
  bool bNoMore;
};

class ORBextractor
{
 public:
  typedef std::shared_ptr<ORBextractor> Ptr;

  enum
  {
    HARRIS_SCORE = 0,
    FAST_SCORE = 1
  };

  ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST,
               int minThFAST);

  // ~ORBextractor(){}

  // unused
  void DetectAndCompute(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors);

  void CalcDescriptors(cv::InputArray image, const std::vector<cv::KeyPoint> &keypoints,
                       cv::OutputArray descriptors);

  // only detect the keypoints in one octave, not compute the descriptors
  void Detect(cv::InputArray image, cv::InputArray mask,
              std::vector<cv::KeyPoint> &keypoints);

  /// only detect the keypoints in pyramid, not compute the descriptors unused
  void DetectWithPyramid(cv::InputArray _image, cv::InputArray _mask,
                         std::vector<cv::KeyPoint> &_keypoints);

  /** select good keypoints from the input keypoints
   * (remove the keypoints which are not FAST corner or beyond borders)
   * compute good keypoints' orientation and size
   */
  void ScreenAndComputeKPsParams(cv::InputArray _image,
                                 std::vector<cv::KeyPoint> &_keypoints,
                                 std::vector<cv::KeyPoint> &out_keypoints);

  int inline GetLevels() { return nlevels; }

  float inline GetScaleFactor() { return scaleFactor; }

  std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

  std::vector<float> inline GetInverseScaleFactors() { return mvInvScaleFactor; }

  std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

  std::vector<float> inline GetInverseScaleSigmaSquares() { return mvInvLevelSigma2; }

  std::vector<cv::Mat> mvImagePyramid;
  std::vector<cv::Mat> mvMaskPyramid;

 protected:
  void ComputePyramid(cv::Mat image, cv::Mat mask);
  void ComputePyramid(cv::Mat image);
  void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);
  std::vector<cv::KeyPoint>
  DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                    const int &maxX, const int &minY, const int &maxY,
                    const int &nFeatures, const int &level);

  void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);
  std::vector<cv::Point> pattern;

  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST;

  std::vector<int> mnFeaturesPerLevel;

  std::vector<int> umax;

  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;
};

} // namespace ssvio

#endif //SSVIO_ORBEXTRACTOR_HPP
