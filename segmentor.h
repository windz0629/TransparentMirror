/*
 *
 */

#ifndef MIRROR_SEGMENTOR_H_
#define MIRROR_SEGMENTOR_H_

#include <QString>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace radi
{
  class Segmentor
  {
    public:
      Segmentor ();
      ~Segmentor ();

      void
      setRawPointCloud (const QString & file_raw_scene);

      void
      setRawPointCloud (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr raw_point_cloud);

      void
      setPassThroughLimitsX (float minimum, float maximum);

      void
      setPassThroughLimitsY (float minimum, float maximum);

      void
      setPassThroughLimitsZ (float minimum, float maximum);

      void
      setRemoveOutlierMeanK (std::size_t mean_k);

      void
      setRemoveOutlierStddev (float stddev);

      void
      setDownsampleStatus (bool status);

      void
      setDownsampleLeafSize (float leaf_x, float leaf_y, float leaf_z);

      void
      setClusterSizes (std::size_t minimum, std::size_t maximum);

      void
      setClusterTolerance (float cluster_tolerance);

      void
      segment (std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cluster_list);

    private:
      QString file_raw_scene_;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_point_cloud_;
      float pass_through_minimum_x_;
      float pass_through_minimum_y_;
      float pass_through_minimum_z_;
      float pass_through_maximum_x_;
      float pass_through_maximum_y_;
      float pass_through_maximum_z_;
      std::size_t remove_outlier_mean_k_;
      float remove_outlier_stddev_;
      bool downsample_status_;
      float downsample_leaf_size_x_;
      float downsample_leaf_size_y_;
      float downsample_leaf_size_z_;
      std::size_t cluster_minimum_;
      std::size_t cluster_maximum_;
      float cluster_tolerance_;

  }; // class Segmentor

} // namespace radi

#endif
