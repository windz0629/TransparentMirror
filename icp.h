/*
 * Iterative Closest Point (ICP) algorithm.
 */

#ifndef RADI_ICF_H_
#define RADI_ICF_H_

#include <string>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <eigen3/Eigen/Dense>

namespace radi
{
  class IterativeClosestPoint
  {
    public:
      // typedefs
      typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
      typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
      typedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;

      IterativeClosestPoint ();
      IterativeClosestPoint (const PointCloudConstPtr & refer_point_cloud, const PointCloudConstPtr & scene_point_cloud);
      ~IterativeClosestPoint ();

      void
      setReferCloud (const PointCloudConstPtr & refer_point_cloud);

      void
      setSceneCloud (const PointCloudConstPtr & scene_point_cloud);

      /*!
       * \fn void setIndices (const IndicesConstPtr & indices)
       * \brief Set the indices of the points which will be involved in ICF algorithm. It can accelerate the
       * computation of ICF algorithm and is usually called during the coarse recognition. When 'indices' is empty,
       * all the points will be used in ICF algorithm.
       * \param[in] indices The indices of the points which will be involved in ICF algorithm.
       */
      void
      setIndices (const IndicesConstPtr & indices);

      void
      setDeviationTranslation (const Eigen::Vector3f & translation_deviation);

      void
      setDeviationRotation (const Eigen::Vector3f & rotation_deviation);

      void
      setIterationOuter (int iteration_outer);

      void
      setIterationInner (int iteration_inner);

      void
      setInitialTransformation (const Eigen::Matrix4f & init_transf);

      void
      setThresholdDistance (float threshold_distance);

      void
      setThresholdAverageDistance (float threshold_average_distance);

      PointCloudConstPtr
      getReferPointCloud () const;

      PointCloudConstPtr
      getScenePointCloud () const;

      void
      getDeviationTranslation (Eigen::Vector3f & translation_deviation);

      void
      getDeviationRotation (Eigen::Vector3f & rotation_deviation);

      int
      getIterationOuter ();

      int
      getIterationInner ();

      const Eigen::Matrix4f
      getInitialTransformation ();

      void
      getThresholdDistance (float & threshold_distance);

      void
      getThresholdAverageDistance (float & threshold_average_distance);

      float
      calObjectiveValue (const Eigen::Matrix4f & mat_transf);

      bool
      estimate (Eigen::Matrix4f & estimated_transf);

    private:
      PointCloudConstPtr refer_point_cloud_;
      PointCloudConstPtr scene_point_cloud_;
      PointCloudConstPtr point_cloud_used_;
      IndicesConstPtr indices_;

      Eigen::Vector3f translation_deviation_;
      Eigen::Vector3f rotation_deviation_;

      std::size_t iteration_outer_;
      std::size_t iteration_inner_;
      Eigen::Matrix4f init_transf_;
      float threshold_distance_;
      float distance_limit_;
      float threshold_average_distance_;
  }; // class IterativeClosestPoint

  const Eigen::Vector3f
  uniformRandom (const Eigen::Vector3f & min_boundary, const Eigen::Vector3f & max_boundary);
  const Eigen::Vector3f
  gaussianRandom (const Eigen::Vector3f & mean, const Eigen::Vector3f & deviation);
  const Eigen::Vector3f
  gaussianRandom(const Eigen::Vector3f & mean, float deviation);

  const Eigen::Vector3f
  matrix2euler (const Eigen::Matrix3f & mat_rotation);
  const Eigen::Matrix3f
  euler2matrix (const Eigen::Vector3f & euler_angle);

} // namespace radi

#endif
