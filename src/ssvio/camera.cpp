//
// Created by weihao on 23-8-9.
//

#include "ssvio/camera.hpp"

namespace ssvio {

Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
{
  return pose_ * T_c_w * p_w;
}

Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w)
{
  return T_c_w.inverse() * pose_inv_ * p_c;
}

Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c)
{
  return Eigen::Vector2d(fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
                         fy_ * p_c(1, 0) / p_c(2, 0) + cy_);
}

Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth)
{
  return Eigen::Vector3d((p_p(0, 0) - cx_) / fx_ * depth,
                         (p_p(1, 0) - cy_) / fy_ * depth,
                         depth);
}

Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w,
                                    double depth)
{
  return camera2world(pixel2camera(p_p, depth), T_c_w);
}

Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
{
  return camera2pixel(world2camera(p_w, T_c_w));
}

void Camera::UndistortImage(cv::Mat &src, cv::Mat &dst)
{
  cv::Mat distortImg = src.clone();

  cv::Mat K_cv = cv::Mat::zeros(3, 3, CV_32F);
  K_cv.at<float>(0, 0) = fx_;
  K_cv.at<float>(0, 2) = cx_;
  K_cv.at<float>(1, 1) = fy_;
  K_cv.at<float>(1, 2) = cy_;
  K_cv.at<float>(2, 2) = 1.0;

  cv::undistort(distortImg, dst, K_cv, dist_coef_);
}
} // namespace ssvio