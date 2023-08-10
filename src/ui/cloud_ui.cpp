//
// Created by weihao on 23-8-10.
//

#include "ui/cloud_ui.hpp"

namespace ui {

std::vector<Eigen::Vector4f> CloudUI::intensity_color_table_pcl_;

CloudUI::CloudUI(const Eigen::Vector3d color, const CloudUI::UseColor use_color_style)
  : use_color_(use_color_style), color_(color.cast<float>())
{
  if (intensity_color_table_pcl_.empty())
  {
    BuildIntensityTable();
  }
}

void CloudUI::AddCloudPoint(const Eigen::Vector3d &point)
{
  if (xyz_data_.size() > max_matin_point_num_)
  {
    LOG(WARNING) << "Cloud pts too many: " << xyz_data_.size() << " delete half pts: ";
    xyz_data_.erase(xyz_data_.begin(), xyz_data_.begin() + xyz_data_.size() / 5);
  }
  xyz_data_.emplace_back(point.cast<float>());
  color_data_height_.push_back(IntensityToRgbPCL(point.z() * 100));
}

void CloudUI::Render()
{
  glPointSize(2);
  glBegin(GL_POINTS);
  for (size_t i = 0; i < xyz_data_.size() && i < color_data_height_.size(); i++)
  {
    auto pos = xyz_data_[i];
    auto color = color_data_height_[i];
    glColor3f(color[0], color[1], color[2]);
    glVertex3d(pos[0], pos[1], pos[2]);
  }
  glEnd();
}

void CloudUI::BuildIntensityTable()
{
  intensity_color_table_pcl_.reserve(255 * 6);
  auto make_color = [](int r, int g, int b) -> Eigen::Vector4f {
    return Eigen::Vector4f(r / 255.0f, g / 255.0f, b / 255.0f, 0.2f);
  };
  for (int i = 0; i < 256; i++)
  {
    intensity_color_table_pcl_.emplace_back(make_color(255, i, 0));
  }
  for (int i = 0; i < 256; i++)
  {
    intensity_color_table_pcl_.emplace_back(make_color(255 - i, 0, 255));
  }
  for (int i = 0; i < 256; i++)
  {
    intensity_color_table_pcl_.emplace_back(make_color(0, 255, i));
  }
  for (int i = 0; i < 256; i++)
  {
    intensity_color_table_pcl_.emplace_back(make_color(255, 255 - i, 0));
  }
  for (int i = 0; i < 256; i++)
  {
    intensity_color_table_pcl_.emplace_back(make_color(i, 0, 255));
  }
  for (int i = 0; i < 256; i++)
  {
    intensity_color_table_pcl_.emplace_back(make_color(0, 255, 255 - i));
  }
}

Eigen::Vector4f CloudUI::IntensityToRgbPCL(const float &intensity) const
{
  int index = int(intensity * 6);
  index = index % intensity_color_table_pcl_.size();
  return intensity_color_table_pcl_[index];
}

} // namespace ui