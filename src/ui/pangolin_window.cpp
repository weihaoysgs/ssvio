//
// Created by weihao on 23-8-7.
//

#include "ui/pangolin_window.hpp"

namespace ui {
PangolinWindow::PangolinWindow() { pangolin_win_impl_ = std::make_unique<PangolinWindowImpl>(); }

bool PangolinWindow::Init()
{
  bool inited = pangolin_win_impl_->InitPangolin();
  if (inited)
  {
    pangolin_win_impl_->render_thread_ = std::thread([this]() { pangolin_win_impl_->Render(); });
  }
  return inited;
}

bool PangolinWindow::ShouldQuit() const { return pangolin::ShouldQuit(); }

PangolinWindow::~PangolinWindow() { Quit(); }

void PangolinWindow::Quit() const
{
  if (pangolin_win_impl_->render_thread_.joinable())
  {
    pangolin_win_impl_->exit_flag_.store(true);
    pangolin_win_impl_->render_thread_.join();
  }
}

void PangolinWindow::ViewImage(const cv::Mat &img_left, const cv::Mat &img_right)
{
  pangolin_win_impl_->SetViewImage(img_left, img_right);
}

void PangolinWindow::PlotAngleValue(float yaw, float pitch, float roll)
{
  pangolin_win_impl_->SetEulerAngle(yaw, pitch, roll);
}

void PangolinWindow::ShowVisualOdomResult(const Sophus::SE3d &pose)
{
  pangolin_win_impl_->UpdateVisualOdometerState(pose);
}
} // namespace ui