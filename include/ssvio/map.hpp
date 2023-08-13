//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_MAP_HPP
#define SSVIO_MAP_HPP

#include "Eigen/Core"
#include "memory"
#include "mutex"
#include "list"

namespace ssvio {

class Feature;
class MapPoint;
class KeyFrame;

class Map
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef std::shared_ptr<Map> Ptr;
  typedef std::unordered_map<unsigned long, std::shared_ptr<KeyFrame>> KeyFramesType;
  typedef std::unordered_map<unsigned long, std::shared_ptr<MapPoint>> MapPointsType;

  Map();

  /** insert new keyframe to the map and the active keyframes
    * insert new KF's mappoints to active mappoints
    * remove the old active KFs and old active mappoints
    */
  void InsertKeyFrame(std::shared_ptr<KeyFrame> kf);

  /// remove mappoints which are not observed by any active kf
  void RemoveOldActiveMapPoints();

  /// remove old keyframes from the active keyframes
  void RemoveOldActiveKeyframe();

  /// insert new mappoint to the map and the active map
  void InsertMapPoint(std::shared_ptr<MapPoint> map_point);

  void InsertActiveMapPoint(std::shared_ptr<MapPoint> map_point);

  void RemoveMapPoint(std::shared_ptr<MapPoint> mappoint);

  /**
   * add the outlier mappoint to a list
   * the mappoints in the list will be removed from the map by RemoveAllOutlierMapPoints()
   */
  void AddOutlierMapPoint(unsigned long mpId);

  void RemoveAllOutlierMapPoints();

  MapPointsType GetAllMapPoints();

  KeyFramesType GetAllKeyFrames();

  MapPointsType GetActiveMapPoints();

  KeyFramesType GetActiveKeyFrames();

 public:
  /// avoid the conflict among different threads' operations on
  /// keyframe's poses and mappoints' positions.
  std::mutex mmutex_map_update_;

 private:
  std::mutex mmutex_data_;
  std::mutex _mmutexOutlierMapPoint;

  MapPointsType all_map_points_;
  MapPointsType activate_map_points_;

  KeyFramesType all_key_frames_;
  KeyFramesType all_active_key_frames_;

  std::list<unsigned long> list_outlier_map_points_;
  std::shared_ptr<KeyFrame> current_keyframe_ = nullptr;
  unsigned int num_active_key_frames_;
};
} // namespace ssvio

#endif //SSVIO_MAP_HPP
