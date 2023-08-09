//
// Created by weihao on 23-8-9.
//

#include "ssvio/orbextractor.hpp"

using namespace cv;
using namespace std;
namespace ssvio {

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

static float IC_Angle(const Mat &image, Point2f pt, const vector<int> &u_max)
{
  int m_01 = 0, m_10 = 0;

  const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

  // Treat the center line differently, v=0
  for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)

    m_10 += u * center[u];

  // Go line by line in the circuI853lar patch
  int step = (int)image.step1();
  for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
  {
    // Proceed over the two lines
    int v_sum = 0;
    int d = u_max[v];
    for (int u = -d; u <= d; ++u)
    {
      int val_plus = center[u + v * step], val_minus = center[u - v * step];
      v_sum += (val_plus - val_minus);
      m_10 += u * (val_plus + val_minus);
    }
    m_01 += v * v_sum;
  }

  return fastAtan2((float)m_01, (float)m_10);
}

const float factorPI = (float)(CV_PI / 180.f);
static void computeOrbDescriptor(const KeyPoint &kpt, const Mat &img,
                                 const Point *pattern, uchar *desc)
{
  float angle = (float)kpt.angle * factorPI;
  float a = (float)cos(angle), b = (float)sin(angle);

  const uchar *center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
  const int step = (int)img.step;

#define GET_VALUE(idx)                                                                   \
  center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step +                       \
         cvRound(pattern[idx].x * a - pattern[idx].y * b)]

  for (int i = 0; i < 32; ++i, pattern += 16)
  {
    int t0, t1, val;
    t0 = GET_VALUE(0);
    t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2);
    t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4);
    t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6);
    t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8);
    t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10);
    t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12);
    t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14);
    t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    desc[i] = (uchar)val;
  }

#undef GET_VALUE
}

// -----------------------------------------------------------------------------

static void makeOffsets(int pixel[25], int rowStride, int patternSize)
{
  static const int offsets16[][2] = {{0, 3},
                                     {1, 3},
                                     {2, 2},
                                     {3, 1},
                                     {3, 0},
                                     {3, -1},
                                     {2, -2},
                                     {1, -3},
                                     {0, -3},
                                     {-1, -3},
                                     {-2, -2},
                                     {-3, -1},
                                     {-3, 0},
                                     {-3, 1},
                                     {-2, 2},
                                     {-1, 3}};

  const int(*offsets)[2] = offsets16;

  CV_Assert(pixel && offsets);

  int k = 0;
  for (; k < patternSize; k++)
    pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
  for (; k < 25; k++)
    pixel[k] = pixel[k - patternSize];
}

// --------------------------------------------------------------------------------------------

ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                           int _iniThFAST, int _minThFAST)
  : nfeatures(_nfeatures)
  , scaleFactor(_scaleFactor)
  , nlevels(_nlevels)
  , iniThFAST(_iniThFAST)
  , minThFAST(_minThFAST)
{
  mvScaleFactor.resize(nlevels);
  mvLevelSigma2.resize(nlevels);
  mvScaleFactor[0] = 1.0f;
  mvLevelSigma2[0] = 1.0f;
  for (int i = 1; i < nlevels; i++)
  {
    mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
    mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
  }

  mvInvScaleFactor.resize(nlevels);
  mvInvLevelSigma2.resize(nlevels);
  for (int i = 0; i < nlevels; i++)
  {
    mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
    mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
  }

  mvImagePyramid.resize(nlevels);
  mvMaskPyramid.resize(nlevels);

  mnFeaturesPerLevel.resize(nlevels);
  float factor = 1.0f / scaleFactor;
  float nDesiredFeaturesPerScale =
      nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

  int sumFeatures = 0;
  for (int level = 0; level < nlevels - 1; level++)
  {
    mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
    sumFeatures += mnFeaturesPerLevel[level];
    nDesiredFeaturesPerScale *= factor;
  }
  mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

  const int npoints = 512;
  const Point *pattern0 = (const Point *)bit_pattern_31_;
  std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

  //This is for orientation
  // pre-compute the end of a row in a circular patch
  umax.resize(HALF_PATCH_SIZE + 1);

  int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
  int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
  const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
  for (v = 0; v <= vmax; ++v)
    umax[v] = cvRound(sqrt(hp2 - v * v));

  // Make sure we are symmetric
  for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
  {
    while (umax[v0] == umax[v0 + 1])
      ++v0;
    umax[v] = v0;
    ++v0;
  }
}

static bool isFastCorner(cv::Mat &img, cv::KeyPoint &keypoint, int threshold)
{
  int patternSize = 16;
  const int K = patternSize / 2, N = patternSize + K + 1;
  int i, pixel[25];
  makeOffsets(pixel, (int)img.step, patternSize);

  threshold = std::min(std::max(threshold, 0), 255);
  uchar threshold_tab[512];
  for (i = -255; i <= 255; i++)
    threshold_tab[i + 255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

  const uchar *ptr = img.ptr<uchar>(cvRound(keypoint.pt.y)) + cvRound(keypoint.pt.x);

  int v = ptr[0];
  const uchar *tab = &threshold_tab[0] - v + 255;

  int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];
  if (d == 0)
    return false;
  d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
  d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
  d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];
  if (d == 0)
    return false;

  d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
  d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
  d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
  d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

  if (d & 1)
  {
    int vt = v - threshold;
    int count = 0;
    for (int k = 0; k < N; k++)
    {
      int pv = ptr[pixel[k]];
      if (pv < vt)
      {
        if (++count > K)
        {
          return true;
        }
      }
      else
      {
        count = 0;
      }
    }
  }

  if (d & 2)
  {
    int vt = v + threshold;
    int count = 0;
    for (int k = 0; k < N; k++)
    {
      int pv = ptr[pixel[k]];
      if (pv > vt)
      {
        if (++count > K)
        {
          return true;
        }
      }
      else
      {
        count = 0;
      }
    }
  }

  return false;
}

static void computeOrientation(const Mat &image, vector<KeyPoint> &keypoints,
                               const vector<int> &umax)
{
  for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                  keypointEnd = keypoints.end();
       keypoint != keypointEnd;
       ++keypoint)
  {
    keypoint->angle = IC_Angle(image, keypoint->pt, umax);
  }
}

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3,
                               ExtractorNode &n4)
{
  const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
  const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

  //Define boundaries of childs
  n1.UL = UL;
  n1.UR = cv::Point2i(UL.x + halfX, UL.y);
  n1.BL = cv::Point2i(UL.x, UL.y + halfY);
  n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
  n1.vKeys.reserve(vKeys.size());

  n2.UL = n1.UR;
  n2.UR = UR;
  n2.BL = n1.BR;
  n2.BR = cv::Point2i(UR.x, UL.y + halfY);
  n2.vKeys.reserve(vKeys.size());

  n3.UL = n1.BL;
  n3.UR = n1.BR;
  n3.BL = BL;
  n3.BR = cv::Point2i(n1.BR.x, BL.y);
  n3.vKeys.reserve(vKeys.size());

  n4.UL = n3.UR;
  n4.UR = n2.BR;
  n4.BL = n3.BR;
  n4.BR = BR;
  n4.vKeys.reserve(vKeys.size());

  //Associate points to childs
  for (size_t i = 0; i < vKeys.size(); i++)
  {
    const cv::KeyPoint &kp = vKeys[i];
    if (kp.pt.x < n1.UR.x)
    {
      if (kp.pt.y < n1.BR.y)
        n1.vKeys.push_back(kp);
      else
        n3.vKeys.push_back(kp);
    }
    else if (kp.pt.y < n1.BR.y)
      n2.vKeys.push_back(kp);
    else
      n4.vKeys.push_back(kp);
  }

  if (n1.vKeys.size() == 1)
    n1.bNoMore = true;
  if (n2.vKeys.size() == 1)
    n2.bNoMore = true;
  if (n3.vKeys.size() == 1)
    n3.bNoMore = true;
  if (n4.vKeys.size() == 1)
    n4.bNoMore = true;
}

vector<cv::KeyPoint>
ORBextractor::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys,
                                const int &minX, const int &maxX, const int &minY,
                                const int &maxY, const int &N, const int &level)
{
  // Compute how many initial nodes
  const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

  const float hX = static_cast<float>(maxX - minX) / nIni;

  list<ExtractorNode> lNodes;

  vector<ExtractorNode *> vpIniNodes;
  vpIniNodes.resize(nIni);

  for (int i = 0; i < nIni; i++)
  {
    ExtractorNode ni;
    ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
    ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
    ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
    ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
    ni.vKeys.reserve(vToDistributeKeys.size());

    lNodes.push_back(ni);
    vpIniNodes[i] = &lNodes.back();
  }

  //Associate points to childs
  for (size_t i = 0; i < vToDistributeKeys.size(); i++)
  {
    const cv::KeyPoint &kp = vToDistributeKeys[i];
    vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
  }

  list<ExtractorNode>::iterator lit = lNodes.begin();

  while (lit != lNodes.end())
  {
    if (lit->vKeys.size() == 1)
    {
      lit->bNoMore = true;
      lit++;
    }
    else if (lit->vKeys.empty())
      lit = lNodes.erase(lit);
    else
      lit++;
  }

  bool bFinish = false;

  int iteration = 0;

  vector<pair<int, ExtractorNode *>> vSizeAndPointerToNode;
  vSizeAndPointerToNode.reserve(lNodes.size() * 4);

  while (!bFinish)
  {
    iteration++;

    int prevSize = lNodes.size();

    lit = lNodes.begin();

    int nToExpand = 0;

    vSizeAndPointerToNode.clear();

    while (lit != lNodes.end())
    {
      if (lit->bNoMore)
      {
        // If node only contains one point do not subdivide and continue
        lit++;
        continue;
      }
      else
      {
        // If more than one point, subdivide
        ExtractorNode n1, n2, n3, n4;
        lit->DivideNode(n1, n2, n3, n4);

        // Add childs if they contain points
        if (n1.vKeys.size() > 0)
        {
          lNodes.push_front(n1);
          if (n1.vKeys.size() > 1)
          {
            nToExpand++;
            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n2.vKeys.size() > 0)
        {
          lNodes.push_front(n2);
          if (n2.vKeys.size() > 1)
          {
            nToExpand++;
            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n3.vKeys.size() > 0)
        {
          lNodes.push_front(n3);
          if (n3.vKeys.size() > 1)
          {
            nToExpand++;
            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n4.vKeys.size() > 0)
        {
          lNodes.push_front(n4);
          if (n4.vKeys.size() > 1)
          {
            nToExpand++;
            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }

        lit = lNodes.erase(lit);
        continue;
      }
    }

    // Finish if there are more nodes than required features
    // or all nodes contain just one point
    if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
    {
      bFinish = true;
    }
    else if (((int)lNodes.size() + nToExpand * 3) > N)
    {
      while (!bFinish)
      {
        prevSize = lNodes.size();

        vector<pair<int, ExtractorNode *>> vPrevSizeAndPointerToNode =
            vSizeAndPointerToNode;
        vSizeAndPointerToNode.clear();

        sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
        for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
        {
          ExtractorNode n1, n2, n3, n4;
          vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

          // Add childs if they contain points
          if (n1.vKeys.size() > 0)
          {
            lNodes.push_front(n1);
            if (n1.vKeys.size() > 1)
            {
              vSizeAndPointerToNode.push_back(
                  make_pair(n1.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n2.vKeys.size() > 0)
          {
            lNodes.push_front(n2);
            if (n2.vKeys.size() > 1)
            {
              vSizeAndPointerToNode.push_back(
                  make_pair(n2.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n3.vKeys.size() > 0)
          {
            lNodes.push_front(n3);
            if (n3.vKeys.size() > 1)
            {
              vSizeAndPointerToNode.push_back(
                  make_pair(n3.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n4.vKeys.size() > 0)
          {
            lNodes.push_front(n4);
            if (n4.vKeys.size() > 1)
            {
              vSizeAndPointerToNode.push_back(
                  make_pair(n4.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }

          lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

          if ((int)lNodes.size() >= N)
            break;
        }

        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
          bFinish = true;
      }
    }
  }

  // Retain the best point in each node
  vector<cv::KeyPoint> vResultKeys;
  vResultKeys.reserve(nfeatures);
  for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
  {
    vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
    cv::KeyPoint *pKP = &vNodeKeys[0];
    float maxResponse = pKP->response;

    for (size_t k = 1; k < vNodeKeys.size(); k++)
    {
      if (vNodeKeys[k].response > maxResponse)
      {
        pKP = &vNodeKeys[k];
        maxResponse = vNodeKeys[k].response;
      }
    }

    vResultKeys.push_back(*pKP);
  }

  return vResultKeys;
}

// -------------------------------------------------------------------------------------

void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint>> &allKeypoints)
{
  allKeypoints.resize(nlevels);

  const float W = 30;

  for (int level = 0; level < nlevels; ++level)
  {
    const int minBorderX = EDGE_THRESHOLD - 3;
    const int minBorderY = minBorderX;
    const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
    const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

    vector<cv::KeyPoint> vToDistributeKeys;
    vToDistributeKeys.reserve(nfeatures * 10);

    const float width = (maxBorderX - minBorderX);
    const float height = (maxBorderY - minBorderY);

    const int nCols = width / W;
    const int nRows = height / W;
    const int wCell = ceil(width / nCols);
    const int hCell = ceil(height / nRows);

    for (int i = 0; i < nRows; i++)
    {
      const float iniY = minBorderY + i * hCell;
      float maxY = iniY + hCell + 6;

      if (iniY >= maxBorderY - 3)
        continue;
      if (maxY > maxBorderY)
        maxY = maxBorderY;

      for (int j = 0; j < nCols; j++)
      {
        const float iniX = minBorderX + j * wCell;
        float maxX = iniX + wCell + 6;
        if (iniX >= maxBorderX - 6)
          continue;
        if (maxX > maxBorderX)
          maxX = maxBorderX;

        vector<cv::KeyPoint> vKeysCell;
        FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
             vKeysCell,
             iniThFAST,
             true);

        if (vKeysCell.empty())
        {
          FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
               vKeysCell,
               minThFAST,
               true);
        }

        if (!vKeysCell.empty())
        {
          for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin();
               vit != vKeysCell.end();
               vit++)
          {
            (*vit).pt.x += j * wCell;
            (*vit).pt.y += i * hCell;
            int h = cvRound((*vit).pt.y);
            int w = cvRound((*vit).pt.x);
            if (mvMaskPyramid[level].ptr<uchar>(h)[w] == 0)
            {
              continue;
            }
            vToDistributeKeys.push_back(*vit);
          }
        }
      }
    }

    vector<KeyPoint> &keypoints = allKeypoints[level];
    keypoints.reserve(nfeatures);

    keypoints = DistributeOctTree(vToDistributeKeys,
                                  minBorderX,
                                  maxBorderX,
                                  minBorderY,
                                  maxBorderY,
                                  mnFeaturesPerLevel[level],
                                  level);

    const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

    // Add border to coordinates and scale information
    const int nkps = keypoints.size();
    for (int i = 0; i < nkps; i++)
    {
      keypoints[i].pt.x += minBorderX;
      keypoints[i].pt.y += minBorderY;
      keypoints[i].octave = level;
      keypoints[i].size = scaledPatchSize;
    }
  }

  // compute orientations
  for (int level = 0; level < nlevels; ++level)
    computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

static void computeDescriptors(const Mat &image, vector<KeyPoint> &keypoints,
                               Mat &descriptors, const vector<Point> &pattern)
{
  descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

  for (size_t i = 0; i < keypoints.size(); i++)
    computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

void ORBextractor::DetectAndCompute(InputArray _image, InputArray _mask,
                                    vector<KeyPoint> &_keypoints,
                                    OutputArray _descriptors)
{
  if (_image.empty())
    return;

  Mat image = _image.getMat();
  Mat mask = _mask.getMat();
  assert(image.type() == CV_8UC1);
  assert(mask.type() == CV_8UC1);

  // Pre-compute the scale pyramid
  ComputePyramid(image, mask);

  vector<vector<KeyPoint>> allKeypoints;
  ComputeKeyPointsOctTree(allKeypoints);

  Mat descriptors;

  int nkeypoints = 0;
  for (int level = 0; level < nlevels; ++level)
    nkeypoints += (int)allKeypoints[level].size();
  if (nkeypoints == 0)
    _descriptors.release();
  else
  {
    _descriptors.create(nkeypoints, 32, CV_8U);
    descriptors = _descriptors.getMat();
  }

  _keypoints.clear();
  _keypoints.reserve(nkeypoints);

  int offset = 0;
  for (int level = 0; level < nlevels; ++level)
  {
    vector<KeyPoint> &keypoints = allKeypoints[level];
    int nkeypointsLevel = (int)keypoints.size();

    if (nkeypointsLevel == 0)
      continue;

    // preprocess the resized image
    Mat workingMat = mvImagePyramid[level].clone();
    GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

    // Compute the descriptors
    Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
    computeDescriptors(workingMat, keypoints, desc, pattern);

    offset += nkeypointsLevel;

    // Scale keypoint coordinates
    if (level != 0)
    {
      float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
      for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                      keypointEnd = keypoints.end();
           keypoint != keypointEnd;
           ++keypoint)
        keypoint->pt *= scale;
    }
    // And add the keypoints to the output
    _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
  }
}

void ORBextractor::Detect(InputArray _image, InputArray _mask,
                          vector<KeyPoint> &_keypoints)
{
  if (_image.empty() || _mask.empty())
    return;

  Mat image = _image.getMat();
  Mat mask = _mask.getMat();
  assert(image.type() == CV_8UC1);
  assert(mask.type() == CV_8UC1);

  const float W = 30;
  const int minBorderX = EDGE_THRESHOLD - 3;
  const int minBorderY = minBorderX;
  const int maxBorderX = image.cols - EDGE_THRESHOLD + 3;
  const int maxBorderY = image.rows - EDGE_THRESHOLD + 3;

  vector<cv::KeyPoint> vToDistributeKeys;
  vToDistributeKeys.reserve(nfeatures * 10);

  const float width = (maxBorderX - minBorderX);
  const float height = (maxBorderY - minBorderY);

  const int nCols = width / W;
  const int nRows = height / W;
  const int wCell = ceil(width / nCols);
  const int hCell = ceil(height / nRows);

  for (int i = 0; i < nRows; i++)
  {
    const float iniY = minBorderY + i * hCell;
    float maxY = iniY + hCell + 6;

    if (iniY >= maxBorderY - 3)
      continue;
    if (maxY > maxBorderY)
      maxY = maxBorderY;

    for (int j = 0; j < nCols; j++)
    {
      const float iniX = minBorderX + j * wCell;
      float maxX = iniX + wCell + 6;
      if (iniX >= maxBorderX - 6)
        continue;
      if (maxX > maxBorderX)
        maxX = maxBorderX;

      vector<cv::KeyPoint> vKeysCell;
      FAST(image.rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, iniThFAST, true);

      if (vKeysCell.empty())
      {
        FAST(image.rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, minThFAST, true);
      }

      if (!vKeysCell.empty())
      {
        for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin();
             vit != vKeysCell.end();
             vit++)
        {
          (*vit).pt.x += j * wCell;
          (*vit).pt.y += i * hCell;
          int h = cvRound((*vit).pt.y);
          int w = cvRound((*vit).pt.x);
          if (mask.ptr<uchar>(h)[w] == 0)
          {
            continue;
          }
          vToDistributeKeys.push_back(*vit);
        }
      }
    }
  }

  _keypoints.reserve(nfeatures);

  _keypoints = DistributeOctTree(
      vToDistributeKeys, minBorderX, maxBorderX, minBorderY, maxBorderY, nfeatures, 0);

  // Add border to coordinates
  const int nkps = _keypoints.size();
  for (int i = 0; i < nkps; i++)
  {
    _keypoints[i].pt.x += minBorderX;
    _keypoints[i].pt.y += minBorderY;
  }
}

void ORBextractor::ScreenAndComputeKPsParams(InputArray _image,
                                             vector<KeyPoint> &_keypoints,
                                             std::vector<cv::KeyPoint> &out_keypoints)
{
  if (_image.empty() || _keypoints.empty())
  {
    LOG(ERROR) << " no image or keypoints to compute the descriptors!";
    return;
  }

  Mat image = _image.getMat();
  assert(image.type() == CV_8UC1);

  out_keypoints.clear();
  out_keypoints.reserve(_keypoints.size());

  ComputePyramid(image);

  for (size_t i = 0, N = _keypoints.size(); i < N; i++)
  {
    cv::KeyPoint &kps = _keypoints[i];
    int level = kps.octave;
    float scale = mvScaleFactor[level];
    cv::Mat imgPyramid = mvImagePyramid[level];

    kps.pt /= scale;

    if (!(kps.pt.y - EDGE_THRESHOLD >= 0 && kps.pt.y + EDGE_THRESHOLD < imgPyramid.rows &&
          kps.pt.x - EDGE_THRESHOLD >= 0 && kps.pt.x + EDGE_THRESHOLD < imgPyramid.cols))
    {
      kps.pt *= scale;
      continue;
    }

    if (!isFastCorner(imgPyramid, kps, minThFAST))
    {
      kps.pt *= scale;
      continue;
    }

    // compute orientation
    kps.angle = IC_Angle(imgPyramid, kps.pt, umax);

    // compute size
    kps.size = PATCH_SIZE * mvScaleFactor[level];

    kps.pt *= scale;

    out_keypoints.push_back(kps);
  }
}

void ORBextractor::DetectWithPyramid(InputArray _image, InputArray _mask,
                                     vector<KeyPoint> &_keypoints)
{
  if (_image.empty() || _mask.empty())
    return;

  Mat image = _image.getMat();
  Mat mask = _mask.getMat();
  assert(image.type() == CV_8UC1);
  assert(mask.type() == CV_8UC1);

  // Pre-compute the scale pyramid
  ComputePyramid(image, mask);

  vector<vector<KeyPoint>> allKeypoints;
  ComputeKeyPointsOctTree(allKeypoints);

  int nkeypoints = 0;
  for (int level = 0; level < nlevels; ++level)
    nkeypoints += (int)allKeypoints[level].size();

  _keypoints.clear();
  _keypoints.reserve(nkeypoints);

  for (int level = 0; level < nlevels; ++level)
  {
    vector<KeyPoint> &keypoints = allKeypoints[level];
    int nkeypointsLevel = (int)keypoints.size();

    if (nkeypointsLevel == 0)
      continue;

    // Scale keypoint coordinates
    if (level != 0)
    {
      float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
      for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                      keypointEnd = keypoints.end();
           keypoint != keypointEnd;
           ++keypoint)
        keypoint->pt *= scale;
    }
    // And add the keypoints to the output
    _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
  }
}

void ORBextractor::CalcDescriptors(InputArray _image, const vector<KeyPoint> &_keypoints,
                                   OutputArray _descriptors)
{
  if (_image.empty() || _keypoints.empty())
  {
    LOG(ERROR) << " no image or keypoints to compute the descriptors!";
    return;
  }

  Mat image = _image.getMat();
  assert(image.type() == CV_8UC1);

  // Pre-compute the scale pyramid
  ComputePyramid(image);
  // prepare the blurred working mat
  std::vector<cv::Mat> vWorkingMats(nlevels);
  for (int level = 0; level < nlevels; level++)
  {
    cv::Mat workingMat = mvImagePyramid[level].clone();
    cv::GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
    vWorkingMats[level] = workingMat;
  }

  cv::Mat descriptors;
  if (_keypoints.size() == 0)
    _descriptors.release();
  else
  {
    _descriptors.release();
    _descriptors.create(_keypoints.size(), 32, CV_8U);
    descriptors = _descriptors.getMat();
  }

  for (size_t i = 0, N = _keypoints.size(); i < N; i++)
  {
    cv::KeyPoint kps = _keypoints[i];
    int level = kps.octave;
    float scale = mvScaleFactor[level];

    cv::Mat desc = descriptors.rowRange(i, i + 1);

    desc = Mat::zeros(1, 32, CV_8UC1);
    kps.pt /= scale;

    computeOrbDescriptor(kps, vWorkingMats[level], &pattern[0], desc.ptr(0));

    kps.pt *= scale;
  }
}

void ORBextractor::ComputePyramid(cv::Mat image, cv::Mat mask)
{
  mvImagePyramid[0] = image.clone();
  mvMaskPyramid[0] = mask.clone();

  for (int level = 0; level < nlevels; ++level)
  {
    float scale = mvInvScaleFactor[level];
    Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));

    // Compute the resized image
    if (level != 0)
    {
      resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
      resize(mvMaskPyramid[level - 1], mvMaskPyramid[level], sz, 0, 0, INTER_LINEAR);
    }
  }
}

void ORBextractor::ComputePyramid(cv::Mat image)
{
  mvImagePyramid[0] = image.clone();

  for (int level = 0; level < nlevels; ++level)
  {
    float scale = mvInvScaleFactor[level];
    Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));

    // Compute the resized image
    if (level != 0)
    {
      resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
    }
  }
}

} // namespace ssvio
