//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_CAMERA_HPP
#define SSVIO_CAMERA_HPP

#include "sophus/se3.hpp"
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
#include "memory"

namespace ssvio {
class Camera
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef std::shared_ptr<Camera> Ptr;

  Camera() = default;
  Camera(double fx, double fy, double cx, double cy, double baseline, const Sophus::SE3d &pose,
         const cv::Mat dist_coef)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose), dist_coef_(dist_coef)
  {
    K_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
  }

  Camera(const Eigen::Matrix3d &K, double baseline, const Sophus::SE3d &pose,
         const cv::Mat &dist_coef)
    : K_(K), baseline_(baseline), pose_(pose), dist_coef_(dist_coef)
  {
    fx_ = K(0, 0);
    fy_ = K(1, 1);
    cx_ = K(0, 2);
    cy_ = K(1, 2);
  }

  Sophus::SE3d getPose() const { return pose_; }
  double getBaseline() const { return baseline_; }
  Eigen::Matrix3d getK() const { return K_; }

  Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w);

  Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w);

  Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c);

  Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, double depth = 1);

  Eigen::Vector3d pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w,
                              double depth = 1);

  Eigen::Vector2d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w);

  void UndistortImage(cv::Mat &src, cv::Mat &dst);

 private:
  double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
  double baseline_ = 0;
  Sophus::SE3d pose_;
  Sophus::SE3d pose_inv_;
  cv::Mat dist_coef_;
  Eigen::Matrix3d K_;
};

} // namespace ssvio

#endif //SSVIO_CAMERA_HPP
