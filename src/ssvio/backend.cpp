//
// Created by weihao on 23-8-10.
//

#include "ssvio/backend.hpp"
#include "ssvio/keyframe.hpp"
#include "ssvio/camera.hpp"
#include "ssvio/feature.hpp"
#include "ssvio/mappoint.hpp"
#include "ssvio/loopclosing.hpp"

namespace ssvio {
Backend::Backend()
{
  _backend_is_running_.store(true);
  _request_pause_.store(false);
  _has_paused_.store(false);

  /// launch the backend thread
  backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::BackendLoop()
{
  while (_backend_is_running_.load())
  {
    /// process all new KFs until the new KF list is empty
    while (CheckNewKeyFrames())
    {
      ProcessNewKeyFrame();
    }

    /// if the loopclosing thread asks backend to pause
    /// this will make sure that the backend will pause in this position, having processed all new KFs in the list
    while (_request_pause_.load())
    {
      _has_paused_.store(true);
      usleep(1000);
    }
    _has_paused_.store(false);

    /// optimize the active KFs and mappoints
    if (!CheckNewKeyFrames() && need_optimization_)
    {
      OptimizeActiveMap();
      need_optimization_ = false; /// this will become true when next new KF is inserted
      // LOG(INFO) << "Backend Running OptimizeActiveMap";
    }

    usleep(1000);
  }
}

void Backend::ProcessNewKeyFrame()
{
  {
    std::unique_lock<std::mutex> lck(_mmutex_new_KF_);
    current_keyframe_ = new_keyfrmaes_.front();
    new_keyfrmaes_.pop_front();
  }

  map_->InsertKeyFrame(current_keyframe_);
  loop_closing_->InsertNewKeyFrame(current_keyframe_);
}

void Backend::InsertKeyFrame(std::shared_ptr<KeyFrame> KF, const bool optimization)
{
  std::unique_lock<std::mutex> lck(_mmutex_new_KF_);
  new_keyfrmaes_.push_back(KF);

  /// need active map optimization when there is a new KF inserted
  need_optimization_ = optimization;
}

void Backend::OptimizeActiveMap()
{
  static auto to_vec2 = [](cv::Point2f &pt) { return Eigen::Vector2d(pt.x, pt.y); };
  typedef g2o::BlockSolver_6_3 BlockSolverType;
  typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  Map::KeyFramesType active_KFs = map_->GetActiveKeyFrames();
  Map::MapPointsType active_MPs = map_->GetActiveMapPoints();

  std::unordered_map<unsigned long, VertexPose *> vertices_kfs;
  unsigned long max_kf_id = 0;
  for (auto &keyframe : active_KFs)
  {
    auto kf = keyframe.second;
    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(kf->key_frame_id_);
    vertex_pose->setEstimate(kf->getPose());
    optimizer.addVertex(vertex_pose);

    max_kf_id = std::max(max_kf_id, kf->key_frame_id_);
    vertices_kfs.insert({kf->key_frame_id_, vertex_pose});
  }

  Eigen::Matrix3d cam_K = camera_left_->getK();
  Sophus::SE3d left_cam_ext_param = camera_left_->getPose();
  Sophus::SE3d right_cam_ext_param = camera_right_->getPose();
  int index = 1;
  double chi2_th = 5.891;

  std::unordered_map<unsigned long, VertexXYZ *> vertices_mps;
  std::unordered_map<EdgeProjection *, Feature::Ptr> edges_and_features;
  for (auto &mappoint : active_MPs)
  {
    MapPoint::Ptr mp = mappoint.second;
    if (mp->is_outlier_)
      continue;

    unsigned long mappoint_id = mp->id_;
    VertexXYZ *v = new VertexXYZ;
    v->setEstimate(mp->getPosition());
    v->setId((max_kf_id + 1) + mappoint_id); /// avoid vertex id equal
    v->setMarginalized(true);

    if (active_KFs.find(
            mp->GetObservations().front().lock()->keyframe_.lock()->key_frame_id_) ==
        active_KFs.end())
    {
      v->setFixed(true);
    }

    vertices_mps.insert({mappoint_id, v});
    optimizer.addVertex(v);

    /// edges
    for (auto &obs : mp->GetActiveObservations())
    {
      Feature::Ptr feat = obs.lock();
      KeyFrame::Ptr kf = feat->keyframe_.lock();

      assert(active_KFs.find(kf->key_frame_id_) != active_KFs.end());
      if (feat->is_outlier_)
        continue;

      EdgeProjection *edge = nullptr;
      if (feat->is_on_left_frame_)
      {
        edge = new EdgeProjection(cam_K, left_cam_ext_param);
      }
      else
      {
        edge = new EdgeProjection(cam_K, right_cam_ext_param);
      }

      edge->setId(index);
      edge->setVertex(0, vertices_kfs.at(kf->key_frame_id_));
      edge->setVertex(1, vertices_mps.at(mp->id_));
      edge->setMeasurement(to_vec2(feat->kp_position_.pt));
      edge->setInformation(Eigen::Matrix2d::Identity());
      auto rk = new g2o::RobustKernelHuber();
      rk->setDelta(chi2_th);
      edge->setRobustKernel(rk);
      edges_and_features.insert({edge, feat});
      optimizer.addEdge(edge);
      index++;
    }
  }

  // do optimization
  int cnt_outlier = 0, cnt_inlier = 0;
  int iteration = 0;

  while (iteration < 5)
  {
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cnt_outlier = 0;
    cnt_inlier = 0;
    // determine if we want to adjust the outlier threshold
    for (auto &ef : edges_and_features)
    {
      if (ef.first->chi2() > chi2_th)
      {
        cnt_outlier++;
      }
      else
      {
        cnt_inlier++;
      }
    }
    double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
    if (inlier_ratio > 0.7)
    {
      break;
    }
    else
    {
      // chi2_th *= 2;
      iteration++;
    }
  }

  // process the outlier edges
  // remove the link between the feature and the mappoint
  for (auto &ef : edges_and_features)
  {
    if (ef.first->chi2() > chi2_th)
    {
      ef.second->is_outlier_ = true;
      auto mappoint = ef.second->map_point_.lock();
      mappoint->RemoveActiveObservation(ef.second);
      mappoint->RemoveObservation(ef.second);
      // if the mappoint has no good observation, then regard it as a outlier. It will be deleted later.
      if (mappoint->GetObservations().empty())
      {
        mappoint->is_outlier_ = true;
        map_->AddOutlierMapPoint(mappoint->id_);
      }
      ef.second->map_point_.reset();
    }
    else
    {
      ef.second->is_outlier_ = false;
    }
  }

  { /// mutex
    std::unique_lock<std::mutex> lck(map_->mmutex_map_update_);
    // update the pose and landmark position
    for (auto &v : vertices_kfs)
    {
      active_KFs.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v : vertices_mps)
    {
      active_MPs.at(v.first)->SetPosition(v.second->estimate());
    }

    // delete outlier mappoints
    map_->RemoveAllOutlierMapPoints();
    map_->RemoveOldActiveMapPoints();
  } // mutex
}

bool Backend::CheckNewKeyFrames()
{
  std::unique_lock<std::mutex> lck(_mmutex_new_KF_);
  return (!new_keyfrmaes_.empty());
}

void Backend::RequestPause()
{
  _request_pause_.store(true);
}

bool Backend::IfHasPaused()
{
  return (_request_pause_.load()) && (_has_paused_.load());
}

void Backend::Resume()
{
  _request_pause_.store(false);
}

void Backend::Stop()
{
  _backend_is_running_.store(false);
  backend_thread_.join();
}

void Backend::SetCameras(std::shared_ptr<Camera> left, std::shared_ptr<Camera> right)
{
  camera_left_ = left;
  camera_right_ = right;
}
} // namespace ssvio