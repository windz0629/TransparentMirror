/*
 * Extract the corner points, points on edges and creases.
 *
 */

#ifndef RADI_FEATURE_EXTRACTOR_H_
#define RADI_FEATURE_EXTRACTOR_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

namespace radi {

  /*!
   * \class FeaturePointExtractor
   * \brief FeaturePointExtractor提取特征上的点，特征包括Corner、Edge和Board。 \par
   *      参考文献： S. Gumhold, X. Wang, and R. S. MacLeod, “Feature Extraction From Point Clouds,” in IMR, 2001.

   */
  class FeaturePointExtractor
  {
    public:
      FeaturePointExtractor ();
      ~FeaturePointExtractor ();

      // typedefs
      typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
      typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

      void
      setInputCloud (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr & point_cloud);

      inline void
      setK (int k) { k_ = k; }

      inline int
      getK () { return (k_); }

      // Obtain the indices of feature points.
      void
      compute ();

      inline const std::vector<int> &
      getCornerPoints () const
      {
        return (corner_point_indices_);
      }

      inline const std::vector<int> &
      getEdgePoints () const
      {
        return (edge_point_indices_);
      }

      const std::vector<int> &
      getBoardPoints () const
      {
        return (board_point_indices_);
      }

      /*!
       * \fn const std::vector<int> getAllFeaturePoints () const
       * \brief 提取所有特征点的index，用于初估变换位姿时减小计算量。
       * \return
       */
      const std::vector<int>
      getAllFeaturePoints () const;

    private:
      PointCloudConstPtr point_cloud_;
      int k_;   /*! Number of points sampled from the neighborhood. */
      std::vector<int> corner_point_indices_;
      std::vector<int> edge_point_indices_;
      std::vector<int> board_point_indices_;
  };

} // namespace radi

#endif // RADI_FEATURE_EXTRACTOR_H_
