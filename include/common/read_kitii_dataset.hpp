//
// Created by weihao on 23-8-9.
//

#ifndef SSVIO_READ_KITII_DATASET_HPP
#define SSVIO_READ_KITII_DATASET_HPP

#include "vector"
#include "fstream"
#include "glog/logging.h"
#include "iomanip"

/// for KITTI gray database
namespace common {
using namespace std;
inline void LoadKittiImagesTimestamps(const string &str_path_to_sequence,
                                      vector<string> &str_image_left_vec_path,
                                      vector<string> &str_image_right_vec_path,
                                      vector<double> &timestamps_vec)
{
  string strPathTimeFile = str_path_to_sequence + "/times.txt";

  std::ifstream fTimes(strPathTimeFile, ios::in | ios::app);

  if (!fTimes.is_open())
    LOG(FATAL) << "Open Failed";
  while (!fTimes.eof())
  {
    string s;
    getline(fTimes, s);
    if (!s.empty())
    {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      timestamps_vec.push_back(t);
    }
    // else
    // {
    //   LOG(ERROR) << "Empty";
    // }
  }

  string strPrefixLeft = str_path_to_sequence + "/image_0/";
  string strPrefixRight = str_path_to_sequence + "/image_1/";

  const size_t nTimes = timestamps_vec.size();
  str_image_left_vec_path.resize(nTimes);
  str_image_right_vec_path.resize(nTimes);

  for (int i = 0; i < nTimes; i++)
  {
    stringstream ss;
    ss << setfill('0') << setw(6) << i;
    str_image_left_vec_path[i] = strPrefixLeft + ss.str() + ".png";
    str_image_right_vec_path[i] = strPrefixRight + ss.str() + ".png";
  }
  fTimes.close();
}
} // namespace common

#endif //SSVIO_READ_KITII_DATASET_HPP
