//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_MAPPOINT_HPP
#define SSVIO_MAPPOINT_HPP

#include "ssvio/frame.hpp"

namespace ssvio {

class MapPoint
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef std::shared_ptr<MapPoint> Ptr;

  MapPoint();
  MapPoint(unsigned long id, Eigen::Vector3d position);

  Eigen::Vector3d getPosition()
  {
    std::unique_lock<std::mutex> lck(update_get_mutex_);
    return position_;
  }

  void SetPosition(const Eigen::Vector3d &position)
  {
    std::unique_lock<std::mutex> lck(update_get_mutex_);
    position_ = position;
  }

  void AddActiveObservation(std::shared_ptr<Feature> feature);
  void AddObservation(std::shared_ptr<Feature> feature);
  std::list<std::weak_ptr<Feature>> GetActiveObservations();
  std::list<std::weak_ptr<Feature>> GetObservations();
  void RemoveActiveObservation(std::shared_ptr<Feature> feat);
  void RemoveObservation(std::shared_ptr<Feature> feat);

 public:
  unsigned long id_ = 0;
  int active_observed_times_ = 0;
  int observed_times_ = 0;
  bool is_outlier_ = false;

 private:
  std::mutex update_get_mutex_;
  std::list<std::weak_ptr<Feature>> active_observations_;
  std::list<std::weak_ptr<Feature>> observations_;
  Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
};
} // namespace ssvio

#endif //SSVIO_MAPPOINT_HPP
