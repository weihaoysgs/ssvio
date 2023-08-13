//
// Created by weihao on 23-8-10.
//

#ifndef SSVIO_CLOUD_UI_HPP
#define SSVIO_CLOUD_UI_HPP

#include "Eigen/Core"
#include "sophus/se3.hpp"
#include "pangolin/pangolin.h"
#include "glog/logging.h"

namespace ui {

class CloudUI
{
 public:
  
  enum UseColor
  {
    SELF_DEFINE_COLOR, /// self define color
    HEIGHT_COLOR,      /// Axis Z height
  };

  CloudUI() = default;
  CloudUI(const Eigen::Vector3d color, const CloudUI::UseColor use_color_style);

  /// 渲染这个点云
  void Render();
  void BuildIntensityTable();
  Eigen::Vector4f IntensityToRgbPCL(const float &intensity) const;
  void AddCloudPoint(const Eigen::Vector3d &point);
  void SetUsedColor(CloudUI::UseColor color) { use_color_ = color; }
  void SetSelfColor(const Eigen::Vector3d &color) { color_ = color.cast<float>(); }

 private:
  const int64_t max_matin_point_num_ = 10000000;
  UseColor use_color_ = UseColor::HEIGHT_COLOR;
  Eigen::Vector3f color_ = Eigen::Vector3f(1, 0, 0);              /// red
  std::vector<Eigen::Vector3f> xyz_data_;                         /// XYZ buffer
  std::vector<Eigen::Vector4f> color_data_height_;                /// color buffer
  static std::vector<Eigen::Vector4f> intensity_color_table_pcl_; /// PCL中intensity table
};

} // namespace ui

#endif //SSVIO_CLOUD_UI_HPP
