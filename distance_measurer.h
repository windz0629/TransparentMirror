// Calculate the distance from a point to the meshes of an stl file.

#ifndef RADI_DISTANCE_MEASURER_H_
#define RADI_DISTANCE_MEASURER_H_

#include <vector>
#include <Eigen/Dense>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace radi {

  class DistanceMeasurer
  {
    public:
      DistanceMeasurer();
      ~DistanceMeasurer();

      void
      setReferPointCloud (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr refer_point_cloud);

      /*!
       * \fn float calShortestDistance (const pcl::PointXYZ & point);
       * \brief Calculate the shortest distance from the specified point to the point cloud.
       * \param[in] point The specified point.
       * \return The shortest distance from the specified point to the point cloud.
       */
      float
      calShortestDistance (const pcl::PointXYZRGB & point);

    private:
      int num_points_;

      int * dev_num_points_;
      float * dev_points_;

  };

  /*!
   * \fn __global__ void distPoint2Point (const float * dev_point, const float * dev_points, const int * dev_num_triangles, float * dev_distances);
   * \brief Calculate the distance between the specified point to each point in the point cloud.
   * \param[in] dev_point Coordinates of the specified point.
   * \param[in] dev_points Coordinates of the points in the point cloud.
   * \param[in] dev_num_points Number of the points in the point cloud.
   * \param[out] dev_distances Ditances from the specified point to each point in the point cloud.
   */
  __global__ void
  distPoint2Point (const float * dev_point, const float * dev_points, const int * dev_num_points, float * dev_distances);

}

#endif
