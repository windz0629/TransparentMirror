/*
 * Extract the corner points, points on edges and creases.
 *
 */

#ifndef RADI_FEATURE_EXTRACTOR_H_
#define RADI_FEATURE_EXTRACTOR_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/gpu/octree/octree.hpp>

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
      typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;

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
      PointCloudPtr point_cloud_;
      pcl::gpu::Octree octree_;
      int k_;   /*! Number of points sampled from the neighborhood. */
      std::vector<int> corner_point_indices_;
      std::vector<int> edge_point_indices_;
      std::vector<int> board_point_indices_;
  };

  // void
  // computeCorrelationMatrices (const pcl::gpu::DeviceArray<pcl::PointXYZ> & dev_point_cloud,
  //     const pcl::gpu::NeighborIndices & dev_neighbor_indices, pcl::gpu::DeviceArray<float> & dev_correlation_matrices);

  void
  computeCorrelationMatricesAndEigenValues (const pcl::gpu::DeviceArray<pcl::PointXYZ> & dev_point_cloud,
      const pcl::gpu::NeighborIndices & dev_neighbor_indices, float * dev_correlation_matrices,
      float * dev_eigen_values);

  __global__ void
  computeCorrelationMatrix(const pcl::PointXYZ * dev_point_cloud_ptr, std::size_t point_number, std::size_t max_elements,
      const int * dev_neighbor_indices_data_ptr, const int * dev_neighbor_indices_sizes_ptr, float * dev_correlation_matrices_ptr);

  __global__ void
  computeEigenValues(const float * dev_correlation_matrices_ptr, std::size_t num_matrices, float * dev_matrix_u,
      float * dev_matrix_s, float * dev_matrix_v, float * dev_eigen_values_ptr);

  // extern __constant__ float SRL_EPS;
  // extern __constant__ float SRL_PI;

  __device__ float srl_sign (float value);

  __device__ float srl_sqrt(float scalar);

  __device__ float srl_asin(float scalar);

  __device__ float srl_acos(float scalar);

  __device__ void srl_svd (const float * matrix_a, int num_dim, float * matrix_u, float * matrix_s, float * matrix_v, float tolerance);

  __device__ void srl_matrix_identity (float * mat_a, int num_row, int num_column);

  __device__ void srl_matrix_zeros (float * mat_a, int num_row, int num_column);

  __device__ void srl_matrix_ones (float * mat_a, int num_row, int num_column);

  __device__ void srl_matrix_copy (const float * mat_src, int length , float * mat_dst);

  __device__ void srl_matrix_add (const float * mat_a, const float * mat_b, int num_row, int num_column, float * mat_c);

  __device__ void srl_matrix_subtract (const float * mat_a, const float * mat_b, int num_row, int num_column, float * mat_c);

  __device__ void srl_matrix_multiply (const float * mat_a, const float * mat_b, int num_row_a, int num_mid, int num_col_b, float * mat_c);

  __device__ void srl_matrix_multiplied_by_scalar (const float * mat_a, int num_row, int num_column, float scalar, float * mat_b);

  __device__ void srl_matrix_divided_by_scalar (const float * mat_a, int num_row, int num_column, float scalar, float * mat_b);

  __device__ void srl_matrix_transpose (const float * mat_a, int num_row, int num_column, float * mat_t);

  __device__ float srl_dot_product (const float * mat_a, const float * mat_b, int length);


} // namespace radi

#endif // RADI_FEATURE_EXTRACTOR_H_
