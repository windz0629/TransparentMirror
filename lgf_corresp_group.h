/*
 * Correspondence group of Local Geometric Features.
 */

#ifndef RADI_LGF_CORRESP_GROUP_H_
#define RADI_LGF_CORRESP_GROUP_H_

#include <vector>
#include <eigen3/Eigen/Dense>
#include <pcl/recognition/cg/correspondence_grouping.h>

#include "local_geometric_feature.h"
#include "icp.h"

namespace radi
{
  class LGFCorrespGroup
  {
    public:
      LGFCorrespGroup ();
      ~LGFCorrespGroup ();

      // typedefs
      typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
      typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

      typedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;

      void
      setReferenceCloud (const PointCloudConstPtr & reference_point_cloud);

      void
      setSceneCloud (const PointCloudConstPtr & scene_point_cloud);

      void
      setFeatureIndices (const IndicesConstPtr & feature_point_indices);

      void
      setReferenceFeatures (const std::vector<CornerEdgesFeature> * reference_corner_features,
          const std::vector<CircleNormalFeature> * reference_circle_features);

      void
      setSceneFeatures (const std::vector<CornerEdgesFeature> * scene_corner_features,
          const std::vector<CircleNormalFeature> * scene_circle_features);

      bool
      recognize (Eigen::Matrix4f & transf_matrix);

    private:
      PointCloudConstPtr reference_point_cloud_;    /*!<! Input point cloud. */
      PointCloudConstPtr scene_point_cloud_;    /*!<! Input point cloud. */
      IndicesConstPtr feature_point_indices_;   /*!<! Indices of the points used in ICF algorithm. */
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_point_cloud_;
      const std::vector<CornerEdgesFeature> * reference_corner_features_;
      const std::vector<CornerEdgesFeature> * scene_corner_features_;
      const std::vector<CircleNormalFeature> * reference_circle_features_;
      const std::vector<CircleNormalFeature> * scene_circle_features_;

      std::vector<Eigen::Matrix4f> transform_candidates_;
      float objective_value_;
      float rate_threshold_;
      Eigen::Matrix4f rough_transform_;

      float angle_threshold_;   /*!< 角度偏差允许值。 */
      float distance_threshold_;    /*!< 判断变换矩阵相同的距离阈值。 */
      float radius_threshold_;      /*!< 半径阈值，用于判断CNF是否匹配。 */
      float resolution_;        /*!< 自旋转的分辨率。 */

      bool
      pairFeatures (const std::vector<float> & angle_seq_scene, const std::vector<float> & angle_seq_refer,
          std::vector<std::vector<pcl::Correspondence> > & angle_corresps_list);

      bool
      hasObtained (const Eigen::Matrix4f & mat_transform);

      bool
      rectifyCandidates ();
  };

} // namespace radi

#endif  // RADI_LGF_CORRESP_GROUP_H_
