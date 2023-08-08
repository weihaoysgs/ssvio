//
// Created by weihao on 23-8-7.
//

#ifndef SSVIO_PANGOLIN_WINDOW_HPP
#define SSVIO_PANGOLIN_WINDOW_HPP

#include "ui/pangolin_window_impl.hpp"

namespace ui {

class PangolinWindow
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  PangolinWindow();
  ~PangolinWindow();

  /// @brief 初始化窗口，后台启动render线程。
  /// @note 与opengl/pangolin无关的初始化，尽量放到此函数体中;
  ///       opengl/pangolin相关的内容，尽量放到PangolinWindowImpl::Init中。
  bool Init();
  bool ShouldQuit() const;
  void Quit() const;
  void ViewImage(const cv::Mat &img_left, const cv::Mat &img_right);
  void PlotAngleValue(float yaw, float pitch, float roll);

  private:
  std::unique_ptr<ui::PangolinWindowImpl> pangolin_win_impl_;
};

} // namespace ui

#endif //SSVIO_PANGOLIN_WINDOW_HPP
