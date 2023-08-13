//
// Created by weihao on 23-8-7.
//

#ifndef SSVIO_PANGOLIN_WINDOW_HPP
#define SSVIO_PANGOLIN_WINDOW_HPP

#include "ui/pangolin_window_impl.hpp"
#include "ssvio/frame.hpp"
#include "ssvio/map.hpp"

namespace ui {

class PangolinWindow
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  PangolinWindow();
  ~PangolinWindow();

  bool Init();
  bool ShouldQuit() const;
  void Quit() const;
  void ViewImage(const cv::Mat &img_left, const cv::Mat &img_right);
  void PlotAngleValue(float yaw, float pitch, float roll);
  void ShowVisualOdomResult(const Sophus::SE3d &pose);
  void AddCurrentFrame(const std::shared_ptr<ssvio::Frame> &frame);
  void AddShowPointCloud(const Eigen::Vector3d &point);
  void SetMap(const std::shared_ptr<ssvio::Map> map) { pangolin_win_impl_->SetMap(map); }
  void SaveTrajectoryAsTUM() { pangolin_win_impl_->SaveTrajectoryTUM(); }

 private:
  std::unique_ptr<ui::PangolinWindowImpl> pangolin_win_impl_;
};

} // namespace ui

#endif //SSVIO_PANGOLIN_WINDOW_HPP
