/*
 * Estimate Corner Edges Feature and Cirecle Normal Feature.
 *
 */

#ifndef RADI_LGF_ESTIMATOR_H_
#define RADI_LGF_ESTIMATOR_H_

#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <Eigen/Core>

#include "local_geometric_feature.h"

namespace radi
{
  struct ClosestPoints
  {
      int idx_line_1;
      int idx_line_2;
      float distance;
      Eigen::Vector3f point_1;
      Eigen::Vector3f point_2;
  };

  /*!
   * \class CEFEstimator
   * \brief
   * \details
   */
  class CEFEstimator
  {
    public:
      CEFEstimator();
      ~CEFEstimator();

      // typedefs
      typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
      typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

      void
      setInputCloud (const PointCloudConstPtr & point_cloud);

      void
      setCornerIndices (const std::vector<int> & corner_indices);

      void
      setFeaturePointIndices (const std::vector<int> & feature_point_indices);

      void
      setRadius (float radius);

      void
      setDistance (float distance);

      void
      setMinNumEdges (std::size_t min_num_edges);

      void
      esimate(std::vector<CornerEdgesFeature> & corner_edges_feature_list);

      float
      getRadius ();

      float
      getMinDistance ();

      std::size_t
      getMinNumEdges ();

    private:
      PointCloudConstPtr point_cloud_;    /*!< 点云。 */
      std::vector<int> corner_indices_;   /*!< 角点Corner的索引。 */
      std::vector<int> feature_point_indices_;  /*!< 特征点，即边界和棱边点，的索引。 */
      float radius_;    /*!< 在角点Corner附近取点的半径。 */
      float distance_;
      int min_num_points_;      /*!< Minimum number of points on a line. */
      int min_num_edges_;   /*!< 生成一个特征所需的最小的边的数目。 */
  };

  /*!
   * \class CNFEstimator
   * \brief
   * \details
   */
  class CNFEstimator
  {
    public:
      CNFEstimator();
      ~CNFEstimator();

      // typedefs
      typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
      typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

      void
      setInputCloud (const PointCloudConstPtr & point_cloud);

      void
      setFeaturePointIndices (const std::vector<int> & feature_point_indices);

      void
      setRadiusLimits (float min_radius, float max_radius);

      void
      setSearchRadius (float search_radius);

      void
      setMinNumPoints (int min_num_points);

      void
      setIdentityThresholds(float position, float radius, float angle);

      void
      esimate (std::vector<CircleNormalFeature> & circle_normal_feature_list);

      float
      getMinRadius ();

      std::size_t
      getMinNumPoints ();

      /*!
       * \fn const pcl::IndicesConstPtr getIndices ()
       * \brief Get the indices of the points in features, such as neighbors of the corner, boad points, etc.
       */
      const pcl::IndicesConstPtr
      getIndices ();

    private:
      PointCloudConstPtr point_cloud_;    /*!< Input point cloud. */
      std::vector<int> feature_point_indices_;
      float min_radius_;    /*!< Minimum radius of the 3D circle, for 3D circle RANSAC Segmentation. */
      float max_radius_;    /*!< Maximum radius of the 3D circle, for 3D circle RANSAC Segmentation. */
      float threshold_distance_;    /*!< Distance threshold for 3D circle RANSAC Segmentation. */
      float search_radius_;     /*!< Search radius during SacSegmentation. */
      int min_num_points_;      /*!< Minimum number of points on one circle. */
      float threshold_position_;    /*!< Threshold of position error which is used to detect whether 2 circles are identical. */
      float threshold_radius_;    /*!< Threshold of position error which is used to detect whether 2 circles are identical. */
      float threshold_angle_;   /*!< Threshold of normal error which is used to detect whether 2 circles are identical. */
      pcl::IndicesConstPtr indices_;  /*!< Indices of the points in features. */


      bool
      isInCircleList(const pcl::ModelCoefficients & coefficients, const std::vector<pcl::ModelCoefficients> & circle_list);
  };


} // namespace radi

#endif // RADI_LGF_ESTIMATOR_H_
