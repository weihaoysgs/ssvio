//
// Created by weihao on 23-8-8.
//

#include "ui/trajectory_ui.hpp"

namespace ui {
void TrajectoryUI::AddTrajectoryPose(const Sophus::SE3d &pose)
{
  pose_.emplace_back(pose.cast<float>());
  if (pose_.size() > max_size_)
  {
    /// Delete half from the begin
    pose_.erase(pose_.begin(), pose_.begin() + pose_.size() / 2);
  }
  position_.emplace_back(pose.translation().cast<float>());
  if (position_.size() > max_size_)
  {
    position_.erase(position_.begin(),
                    position_.begin() + position_.size() / 2);
  }
}

void TrajectoryUI::Render()
{
  if (position_.empty())
  {
    return;
  }
  glPointSize(5);
  glBegin(GL_POINTS);
  for (auto &p : position_)
  {
    glColor3f(color_[0], color_[1], color_[2]);
    glVertex3d(p[0], p[1], p[2]);
  }
  glEnd();
}

void TrajectoryUI::Clear()
{
  position_.clear();
  position_.reserve(max_size_);
}

}  // namespace ui