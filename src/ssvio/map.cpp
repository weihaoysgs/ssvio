//
// Created by weihao on 23-8-9.
//

#include "ssvio/map.hpp"
#include "ssvio/mappoint.hpp"
#include "ssvio/keyframe.hpp"
#include "ssvio/setting.hpp"
#include "ssvio/feature.hpp"

namespace ssvio {

Map::Map()
{
  num_active_key_frames_ = Setting::Get<int>("Map.ActiveMap.Size");
}

void Map::InsertKeyFrame(std::shared_ptr<KeyFrame> kf)
{
  current_keyframe_ = kf;

  {
    std::unique_lock<std::mutex> lck(mmutex_data_);

    // insert keyframe
    if (all_key_frames_.find(kf->key_frame_id_) == all_key_frames_.end())
    {
      all_key_frames_.insert(std::make_pair(kf->key_frame_id_, kf));
      all_active_key_frames_.insert(std::make_pair(kf->key_frame_id_, kf));
    }
    else
    {
      LOG(FATAL) << "KF ID Equal";
      all_key_frames_[kf->key_frame_id_] = kf;
      all_active_key_frames_[kf->key_frame_id_] = kf;
    }
  }

  // add the new KF to its observed mappoints' active observations
  // insert new KF's mappoints to active mappoints
  for (auto &feat : kf->features_left_)
  {
    auto mp = feat->map_point_.lock();
    if (mp)
    {
      mp->AddActiveObservation(feat);
      InsertActiveMapPoint(mp);
    }
  }

  // remove old keyframe and mappoints from the active map
  if (all_active_key_frames_.size() > num_active_key_frames_)
  {
    RemoveOldActiveKeyframe();
    RemoveOldActiveMapPoints();
  }
}

void Map::InsertMapPoint(MapPoint::Ptr map_point)
{
  std::unique_lock<std::mutex> lck(mmutex_data_);

  if (all_map_points_.find(map_point->id_) == all_map_points_.end())
  {
    all_map_points_.insert(make_pair(map_point->id_, map_point));
  }
  else
  {
    LOG(FATAL) << "map_point ID Equal";
    all_map_points_[map_point->id_] = map_point;
  }
}

void Map::InsertActiveMapPoint(MapPoint::Ptr map_point)
{
  std::unique_lock<std::mutex> lck(mmutex_data_);

  if (activate_map_points_.find(map_point->id_) == activate_map_points_.end())
  {
    activate_map_points_.insert(make_pair(map_point->id_, map_point));
  }
  else
  {
    // LOG(FATAL) << "active_map_point ID Equal";
    activate_map_points_[map_point->id_] = map_point;
  }
}

void Map::RemoveOldActiveKeyframe()
{
  std::unique_lock<std::mutex> lck(mmutex_data_);

  if (current_keyframe_ == nullptr)
    return;

  double maxDis = 0, minDis = 9999;
  double maxKFId = 0, minKFId = 0;

  // compute the min distance and max distance between current kf and previous active kfs
  auto Twc = current_keyframe_->getPose().inverse();
  for (auto &kf : all_active_key_frames_)
  {
    if (kf.second == current_keyframe_)
      continue;
    auto dis = (kf.second->getPose() * Twc).log().norm();
    if (dis > maxDis)
    {
      maxDis = dis;
      maxKFId = kf.first;
    }
    else if (dis < minDis)
    {
      minDis = dis;
      minKFId = kf.first;
    }
  }

  // decide which kf to be removed
  const double minDisTh = 0.2;
  KeyFrame::Ptr frame_to_remove = nullptr;
  if (minDis < minDisTh)
  {
    frame_to_remove = all_active_key_frames_.at(minKFId);
  }
  else
  {
    frame_to_remove = all_active_key_frames_.at(maxKFId);
  }

  // remove the kf and its mappoints active observation
  all_active_key_frames_.erase(frame_to_remove->key_frame_id_);
  for (auto &feat : frame_to_remove->features_left_)
  {
    auto mp = feat->map_point_.lock();
    if (mp)
    {
      mp->RemoveActiveObservation(feat);
    }
  }
}

void Map::RemoveOldActiveMapPoints()
{
  // if the mappoint has no active observation, then remove it from the active mappoints
  std::unique_lock<std::mutex> lck(mmutex_data_);

  int cnt_active_landmark_removed = 0;
  for (auto iter = activate_map_points_.begin(); iter != activate_map_points_.end();)
  {
    if (iter->second->active_observed_times_ == 0)
    {
      iter = activate_map_points_.erase(iter);
      cnt_active_landmark_removed++;
    }
    else
    {
      ++iter;
    }
  }
}

void Map::RemoveMapPoint(std::shared_ptr<MapPoint> mappoint)
{
  std::unique_lock<std::mutex> lck(mmutex_data_);

  unsigned long mpId = mappoint->id_;

  // delete from all mappoints
  all_map_points_.erase(mpId);

  // delete from active mappoints
  activate_map_points_.erase(mpId);
}

void Map::AddOutlierMapPoint(unsigned long mpId)
{
  std::unique_lock<std::mutex> lck(_mmutexOutlierMapPoint);
  list_outlier_map_points_.push_back(mpId);
}

void Map::RemoveAllOutlierMapPoints()
{
  std::unique_lock<std::mutex> lck(mmutex_data_);
  std::unique_lock<std::mutex> lck_1(_mmutexOutlierMapPoint);

  for (auto iter = list_outlier_map_points_.begin();
       iter != list_outlier_map_points_.end();
       iter++)
  {
    all_map_points_.erase(*iter);
    activate_map_points_.erase(*iter);
  }
  list_outlier_map_points_.clear();
}

Map::MapPointsType Map::GetAllMapPoints()
{
  std::unique_lock<std::mutex> lck(mmutex_data_);
  return all_map_points_;
}

Map::KeyFramesType Map::GetAllKeyFrames()
{
  std::unique_lock<std::mutex> lck(mmutex_data_);
  return all_key_frames_;
}

Map::MapPointsType Map::GetActiveMapPoints()
{
  std::unique_lock<std::mutex> lck(mmutex_data_);
  return activate_map_points_;
}

Map::KeyFramesType Map::GetActiveKeyFrames()
{
  std::unique_lock<std::mutex> lck(mmutex_data_);
  return all_active_key_frames_;
}

} // namespace ssvio