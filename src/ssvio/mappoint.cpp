//
// Created by weihao on 23-8-9.
//

#include "ssvio/mappoint.hpp"
#include "ssvio/feature.hpp"

namespace ssvio {

MapPoint::MapPoint()
{
  static unsigned long FactoryId = 0;
  id_ = FactoryId++;
}

MapPoint::MapPoint(unsigned long id, Eigen::Vector3d position)
{
  id_ = id;
  position_ = position;
}

void MapPoint::AddObservation(std::shared_ptr<Feature> feature)
{
  std::unique_lock<std::mutex> lck(update_get_mutex_);
  observations_.emplace_back(feature);
  observed_times_++;
}

void MapPoint::AddActiveObservation(std::shared_ptr<Feature> feature)
{
  std::unique_lock<std::mutex> lck(update_get_mutex_);
  active_observations_.emplace_back(feature);
  active_observed_times_++;
}

void MapPoint::RemoveActiveObservation(std::shared_ptr<Feature> feature)
{
  std::unique_lock<std::mutex> lck(update_get_mutex_);
  for (auto iter = active_observations_.begin(); iter != active_observations_.end();
       iter++)
  {
    /// 移除掉该特征点的观测，系统中都是同一块内存，通过指针地址比较即可
    if (iter->lock() == feature)
    {
      active_observations_.erase(iter);
      active_observed_times_--;
      /// break 结束，只有一个
      break;
    }
  }
}

void MapPoint::RemoveObservation(std::shared_ptr<Feature> feature)
{
  std::unique_lock<std::mutex> lck(update_get_mutex_);
  for (auto iter = observations_.begin(); iter != observations_.end(); iter++)
  {
    if (iter->lock() == feature)
    {
      observations_.erase(iter);
      feature->map_point_.reset();
      observed_times_--;
      /// break 结束，只有一个
      break;
    }
  }
}

std::list<std::weak_ptr<Feature>> MapPoint::GetActiveObservations()
{
  std::unique_lock<std::mutex> lck(update_get_mutex_);
  return active_observations_;
}

std::list<std::weak_ptr<Feature>> MapPoint::GetObservations()
{
  std::unique_lock<std::mutex> lck(update_get_mutex_);
  return observations_;
}

} // namespace ssvio
