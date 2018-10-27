#include <numeric>
#include <boost/range/numeric.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/make_shared.hpp>
#include <Eigen/Eigenvalues>

#include <pcl/octree/octree_search.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/filters/extract_indices.h>

#include <math.h>
#include <thrust/device_vector.h>


// Test
#include <pcl/visualization/pcl_visualizer.h>

#include "feature_point_extractor.h"

namespace radi
{

  float
  calMaxIncludedAngle(const std::vector<Eigen::Vector3f> & vectors, const Eigen::Vector3f & normal);

  FeaturePointExtractor::FeaturePointExtractor() : point_cloud_(new pcl::PointCloud<pcl::PointXYZ>()), k_(30),
      octree_(pcl::gpu::Octree()), corner_point_indices_(std::vector<int>()),
      edge_point_indices_(std::vector<int>()), board_point_indices_(std::vector<int>())
  { }

  FeaturePointExtractor::~FeaturePointExtractor()
  { }

  void
  FeaturePointExtractor::setInputCloud (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr & point_cloud)
  {
    pcl::copyPointCloud(*point_cloud, *point_cloud_);
    pcl::gpu::DeviceArray<pcl::PointXYZ> dev_point_array;
    dev_point_array.upload(point_cloud_->points);
    // pcl::gpu::DeviceArray<pcl::PointXYZ> dev_point_array(point_cloud_->size());
    octree_.setCloud(dev_point_array);
    octree_.build();
    if (octree_.isBuilt())
      std::cout << "Octree on GPU is built." << std::endl;
  }

  void
  FeaturePointExtractor::compute ()
  {
    int max_elements = 1000;
    float radius = 0.02;
    pcl::gpu::DeviceArray<pcl::PointXYZ> dev_point_array;
    dev_point_array.upload(point_cloud_->points);
    pcl::gpu::NeighborIndices neighbor_indices (point_cloud_->size(), max_elements);
    octree_.radiusSearch(dev_point_array, radius, max_elements, neighbor_indices);

    std::cout << "Number of points in the cloud: " << point_cloud_->size() << std::endl;
    std::cout << "Size of neighbor_indices.data: " << neighbor_indices.data.size() << std::endl;
    std::vector<int> neighbor_sizes (neighbor_indices.sizes.size());
    neighbor_indices.sizes.download(&(neighbor_sizes[0]));
    std::size_t total_neighbors = 0;
    for (std::size_t i = 0; i < neighbor_indices.sizes.size(); ++i)
      total_neighbors += neighbor_sizes[i];

    std::cout << "Total neighbors: " << total_neighbors << std::endl;

    float * dev_correlation_matrices;
    float * dev_eigen_values;
    cudaMalloc ((void **)&dev_correlation_matrices, (9*point_cloud_->size())*sizeof(float));
    cudaMalloc ((void **)&dev_eigen_values, (3*point_cloud_->size())*sizeof(float));
    computeCorrelationMatricesAndEigenValues(dev_point_array, neighbor_indices, dev_correlation_matrices, dev_eigen_values);

    float * eigen_values = (float *)malloc(3*point_cloud_->size()*sizeof(float));
    cudaMemcpy(eigen_values, dev_eigen_values, 3*point_cloud_->size()*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < point_cloud_->size(); ++i)
        std::cout << eigen_values[3*i+0] << "  " << eigen_values[3*i+1] << "  " << eigen_values[3*i+2] << std::endl;

    cudaFree (dev_correlation_matrices);
    cudaFree (dev_eigen_values);

    /*
    std::vector<Eigen::Vector3f> omega_cr_list (point_cloud_->size ());
    std::vector<float> omega_b_2_list (point_cloud_->size ());
    std::vector<std::vector<int> > neighbor_indices_list (point_cloud_->size ());

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree (0.01);
    octree.setInputCloud (point_cloud_);
    octree.addPointsFromInputCloud ();
    for (std::size_t idx_point = 0; idx_point < point_cloud_->size (); ++idx_point)
    {
      const pcl::PointXYZRGB & point = (*point_cloud_)[idx_point];
      Eigen::Vector3f point_p (point.x, point.y, point.z);
      // Search 'k' points around the 'point'.
      // The first point of the neighborhood is the query point.
      std::vector<int> neighbor_indices;
      std::vector<float> neighbor_distances;
      // octree.nearestKSearch(point, k_, neighbor_indices, neighbor_distances);
      octree.radiusSearch(point, 0.03, neighbor_indices, neighbor_distances);
      neighbor_indices_list[idx_point] = neighbor_indices;

      if (neighbor_indices.size () > 1)
      {
        // Calculate the center location of the neighborhood.
        Eigen::Vector3f center (0.0, 0.0, 0.0);
        for (std::size_t idx_neighbor = 0; idx_neighbor < neighbor_indices.size (); ++idx_neighbor)
        {

          const pcl::PointXYZRGB & point_neighbor = (*point_cloud_)[neighbor_indices[idx_neighbor]];
          Eigen::Vector3f point_q (point_neighbor.x, point_neighbor.y, point_neighbor.z);
          center += point_q;
        }
        center /= neighbor_indices.size();

        // Calculate correlation matrix.
        Eigen::Matrix3f correlation_matrix = Eigen::MatrixXf::Zero (3 ,3);
        for (std::size_t idx_neighbor = 0; idx_neighbor < neighbor_indices.size(); ++idx_neighbor)
        {
          const pcl::PointXYZRGB & point_neighbor = (*point_cloud_)[neighbor_indices[idx_neighbor]];
          Eigen::Vector3f point_q (point_neighbor.x, point_neighbor.y, point_neighbor.z);
          Eigen::Vector3f vect_cq = point_q - center;
          correlation_matrix += vect_cq * Eigen::Transpose<Eigen::Vector3f>(vect_cq);
        }

        // Calculate eigen values and eigen vectors of the correlation matrix.
        Eigen::EigenSolver<Eigen::Matrix3f> eigen_solver;
        eigen_solver.compute(correlation_matrix);
        std::vector<float> eigen_values(3);
        eigen_values[0] = eigen_solver.eigenvalues()[0].real();
        eigen_values[1] = eigen_solver.eigenvalues()[1].real();
        eigen_values[2] = eigen_solver.eigenvalues()[2].real();
        std::vector<Eigen::Vector3f> eigen_vectors(3);
        eigen_vectors[0] = eigen_solver.eigenvectors().col(0).real();
        eigen_vectors[1] = eigen_solver.eigenvectors().col(1).real();
        eigen_vectors[2] = eigen_solver.eigenvectors().col(2).real();

        // Sort the eigen values.
        std::vector<std::size_t> order_indices(3);
        std::iota(order_indices.begin (), order_indices.end (), 0);
        std::sort(order_indices.begin(), order_indices.end(), [&eigen_values](int idx_1, int idx_2)
            { return eigen_values[idx_1] < eigen_values[idx_2]; });
        float lambda_0 = eigen_values[order_indices[0]];
        float lambda_1 = eigen_values[order_indices[1]];
        float lambda_2 = eigen_values[order_indices[2]];
        Eigen::Vector3f eig_vector_0 = eigen_vectors[order_indices[0]];
        Eigen::Vector3f eig_vector_2 = eigen_vectors[order_indices[2]];

        // Calculate penalty function for detecting crease points.
        omega_cr_list[idx_point] = std::max (lambda_1-lambda_0, std::abs (lambda_2-(lambda_1+lambda_0)))
                / lambda_2 * eig_vector_2;

        // Calculate penalty function for detecting board points.
        // Calculate beta.
        std::vector<Eigen::Vector3f> project_vectors(neighbor_indices.size () - 1);
        for (std::size_t idx_neighbor = 1; idx_neighbor < neighbor_indices.size(); ++idx_neighbor)
        {
          const pcl::PointXYZRGB & point_neighbor = (*point_cloud_)[neighbor_indices[idx_neighbor]];
          Eigen::Vector3f point_q (point_neighbor.x, point_neighbor.y, point_neighbor.z);
          Eigen::Vector3f vect_pq = point_q - point_p;
          project_vectors[idx_neighbor-1] = vect_pq - vect_pq.dot(eig_vector_0)/eig_vector_0.dot(eig_vector_0) * eig_vector_0;
        }

        float beta = calMaxIncludedAngle(project_vectors, eig_vector_0);
        omega_b_2_list[idx_point] = 1.0 - beta / (2.0*3.14151926);
      }
    }

    // Detect edge.
    float tau = 0.4;
    for (std::size_t idx_point = 0; idx_point < point_cloud_->size (); ++idx_point)
      if (omega_cr_list[idx_point].norm () < tau)
        edge_point_indices_.push_back(idx_point);

    // Remove duplicate indices.
    std::set<int> edge_point_indices_set(edge_point_indices_.begin(), edge_point_indices_.end());
    edge_point_indices_ = std::vector<int>(edge_point_indices_set.begin(), edge_point_indices_set.end());

    // Detect board.
    tau = 0.7;
    for (std::size_t idx_point = 0; idx_point < point_cloud_->size (); ++idx_point)
      if (omega_b_2_list[idx_point] < tau)
        board_point_indices_.push_back(idx_point);

    // Remove duplicate indices.
    std::set<int> board_point_indices_set(board_point_indices_.begin(), board_point_indices_.end());
    board_point_indices_ = std::vector<int>(board_point_indices_set.begin(), board_point_indices_set.end());

    */
  }

  const std::vector<int>
  FeaturePointExtractor::getAllFeaturePoints () const
  {
    std::size_t raw_total_points = corner_point_indices_.size() + edge_point_indices_.size() + board_point_indices_.size();
    std::vector<int> raw_total_point_indices (raw_total_points);
    std::copy (corner_point_indices_.begin(), corner_point_indices_.end(), raw_total_point_indices.begin());
    std::copy (edge_point_indices_.begin(), edge_point_indices_.end(), raw_total_point_indices.begin()+corner_point_indices_.size());
    std::copy (board_point_indices_.begin(), board_point_indices_.end(),
        raw_total_point_indices.begin()+corner_point_indices_.size()+edge_point_indices_.size());

    // Remove duplicate indices.
    std::set<int> total_point_indices_set(raw_total_point_indices.begin(), raw_total_point_indices.end());
    std::vector<int> total_point_indices_ = std::vector<int>(total_point_indices_set.begin(), total_point_indices_set.end());

    return (total_point_indices_);
  }

  float
  calMaxIncludedAngle (const std::vector<Eigen::Vector3f> & vectors, const Eigen::Vector3f & normal)
  {
    // Choose the first vector as X axis.
    Eigen::Vector3f x_axis = vectors[0] / vectors[0].norm ();
    Eigen::Vector3f y_axis = (normal/normal.norm ()).cross (x_axis);

    std::vector<float> included_angles (vectors.size ());
    for (int idx_vector = 0; idx_vector < vectors.size (); ++idx_vector)
    {
      float x_comp = vectors[idx_vector].dot (x_axis) / x_axis.dot (x_axis);
      float y_comp = vectors[idx_vector].dot (y_axis) / y_axis.dot (y_axis);
      included_angles[idx_vector] = std::atan2 (y_comp, x_comp);
      if (included_angles[idx_vector] < 0)
      {
        included_angles[idx_vector] += 2*3.1415926;
      }
    }

    // Sort the included angles.
    std::vector<std::size_t> order_indices (included_angles.size ());
    std::iota (order_indices.begin (), order_indices.end (), 0);
    std::sort (order_indices.begin(), order_indices.end(), [&included_angles](int idx_1, int idx_2)
        { return included_angles[idx_1] < included_angles[idx_2]; });

    float beta = 0.0;
    for (int idx_order = 0; idx_order < order_indices.size () - 1 ; ++idx_order)
    {
      float angle = included_angles[order_indices[idx_order+1]] - included_angles[order_indices[idx_order]];
      if (angle > beta)
      {
        beta = angle;
      }
    }

    return (beta);
  }

  void
  computeCorrelationMatricesAndEigenValues (const pcl::gpu::DeviceArray<pcl::PointXYZ> & dev_point_cloud,
      const pcl::gpu::NeighborIndices & dev_neighbor_indices, float * dev_correlation_matrices,
      float * dev_eigen_values)
  {
    const pcl::PointXYZ * dev_point_cloud_ptr = dev_point_cloud.ptr();
    const int * dev_neighbor_indices_data_ptr = dev_neighbor_indices.data.ptr();
    const int * dev_neighbor_indices_sizes_ptr = dev_neighbor_indices.sizes.ptr();
    std::size_t point_number = dev_point_cloud.size();
    std::size_t max_elements = dev_neighbor_indices.max_elems;
    // float * dev_correlation_matrices_ptr = dev_correlation_matrices.ptr();
    computeCorrelationMatrix<<<(point_number+255)/256, 256>>> (dev_point_cloud_ptr, point_number, max_elements,
        dev_neighbor_indices_data_ptr, dev_neighbor_indices_sizes_ptr, dev_correlation_matrices);

    float * correlate_matrices = (float *)malloc(9*dev_point_cloud.size()*sizeof(float));
    cudaMemcpy(correlate_matrices, dev_correlation_matrices, 9*dev_point_cloud.size()*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < dev_point_cloud.size(); ++i)
    {
      std::cout << correlate_matrices[9*i+0] << "  " << correlate_matrices[9*i+3] << "  " << correlate_matrices[9*i+6] << std::endl;
      std::cout << correlate_matrices[9*i+1] << "  " << correlate_matrices[9*i+4] << "  " << correlate_matrices[9*i+7] << std::endl;
      std::cout << correlate_matrices[9*i+2] << "  " << correlate_matrices[9*i+5] << "  " << correlate_matrices[9*i+8] << std::endl;
      std::cout << std::endl;
    }

    float * dev_matrix_u;
    float * dev_matrix_s;
    float * dev_matrix_v;
    cudaMalloc ((void **)&dev_matrix_u, (9*point_number)*sizeof(float));
    cudaMalloc ((void **)&dev_matrix_s, (9*point_number)*sizeof(float));
    cudaMalloc ((void **)&dev_matrix_v, (9*point_number)*sizeof(float));
    computeEigenValues<<<(point_number+255)/256, 256>>>(dev_correlation_matrices, point_number, dev_matrix_u,
        dev_matrix_s, dev_matrix_v, dev_eigen_values);

    cudaFree(dev_matrix_u);
    cudaFree(dev_matrix_s);
    cudaFree(dev_matrix_v);
  }

  __global__ void
  computeCorrelationMatrix(const pcl::PointXYZ * dev_point_cloud_ptr, std::size_t point_number, std::size_t max_elements,
      const int * dev_neighbor_indices_data_ptr, const int * dev_neighbor_indices_sizes_ptr, float * dev_correlation_matrices_ptr)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < point_number)
    {
      // ToDo: Calculate omega_cr_list and omega_b_2_list.
      std::size_t neighbor_size = *(dev_neighbor_indices_sizes_ptr+tid);
      const int * dev_neighbor_indices_begin = dev_neighbor_indices_data_ptr + max_elements*tid;
      // thrust::device_vector<float> dev_center(3);
      float dev_center[3];
      dev_center[0] = 0.0;
      dev_center[1] = 0.0;
      dev_center[2] = 0.0;
      for (std::size_t i = 0; i < neighbor_size; ++i)
      {
        dev_center[0] += (dev_point_cloud_ptr+(*(dev_neighbor_indices_begin+i)))->x;
        dev_center[1] += (dev_point_cloud_ptr+(*(dev_neighbor_indices_begin+i)))->y;
        dev_center[2] += (dev_point_cloud_ptr+(*(dev_neighbor_indices_begin+i)))->z;
      }
      dev_center[0] /= static_cast<float> (neighbor_size);
      dev_center[1] /= static_cast<float> (neighbor_size);
      dev_center[2] /= static_cast<float> (neighbor_size);

      // Calculate correlation matrix.
      float * correlation_matrix_begin = dev_correlation_matrices_ptr + 9*tid;
      for (std::size_t i = 0; i < neighbor_size; ++i)
      {
        float dev_vect_cq[3];
        dev_vect_cq[0] = (dev_point_cloud_ptr+(*(dev_neighbor_indices_begin+i)))->x - dev_center[0];
        dev_vect_cq[1] = (dev_point_cloud_ptr+(*(dev_neighbor_indices_begin+i)))->y - dev_center[1];
        dev_vect_cq[2] = (dev_point_cloud_ptr+(*(dev_neighbor_indices_begin+i)))->z - dev_center[2];
        correlation_matrix_begin[0] = dev_vect_cq[0] * dev_vect_cq[0];
        correlation_matrix_begin[1] = dev_vect_cq[1] * dev_vect_cq[0];
        correlation_matrix_begin[2] = dev_vect_cq[2] * dev_vect_cq[0];
        correlation_matrix_begin[3] = dev_vect_cq[0] * dev_vect_cq[1];
        correlation_matrix_begin[4] = dev_vect_cq[1] * dev_vect_cq[1];
        correlation_matrix_begin[5] = dev_vect_cq[2] * dev_vect_cq[1];
        correlation_matrix_begin[6] = dev_vect_cq[0] * dev_vect_cq[2];
        correlation_matrix_begin[7] = dev_vect_cq[1] * dev_vect_cq[2];
        correlation_matrix_begin[8] = dev_vect_cq[2] * dev_vect_cq[2];
      }
    }
  }

  __global__ void
  computeEigenValues(const float * dev_correlation_matrices_ptr, std::size_t num_matrices, float * dev_matrix_u,
      float * dev_matrix_s, float * dev_matrix_v, float * dev_eigen_values_ptr)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < num_matrices)
    {
      float * matrix_u = dev_matrix_u + 9*tid;
      float * matrix_s = dev_matrix_s + 9*tid;
      float * matrix_v = dev_matrix_v + 9*tid;
      srl_svd (dev_correlation_matrices_ptr+9*tid, 3, matrix_u, matrix_s, matrix_v, 1E-6);

      if ((matrix_s[0] < matrix_s[1]) && (matrix_s[0] < matrix_s[2]))
        dev_eigen_values_ptr[3*tid] = matrix_s[0];
      else if ((matrix_s[1] < matrix_s[0]) && (matrix_s[1] < matrix_s[2]))
        dev_eigen_values_ptr[3*tid] = matrix_s[1];
      else
        dev_eigen_values_ptr[3*tid] = matrix_s[2];

      if (((matrix_s[0] > matrix_s[1]) && (matrix_s[0] < matrix_s[2])) || ((matrix_s[0] < matrix_s[1]) && (matrix_s[0] > matrix_s[2])))
        dev_eigen_values_ptr[3*tid+1] = matrix_s[0];
      else if (((matrix_s[1] > matrix_s[0]) && (matrix_s[1] < matrix_s[2])) || ((matrix_s[1] < matrix_s[0]) && (matrix_s[1] > matrix_s[2])))
        dev_eigen_values_ptr[3*tid+1] = matrix_s[1];
      else
        dev_eigen_values_ptr[3*tid+1] = matrix_s[2];

      if ((matrix_s[0] > matrix_s[1]) && (matrix_s[0] > matrix_s[2]))
        dev_eigen_values_ptr[3*tid+2] = matrix_s[0];
      else if ((matrix_s[1] > matrix_s[0]) && (matrix_s[1] > matrix_s[2]))
        dev_eigen_values_ptr[3*tid+2] = matrix_s[1];
      else
        dev_eigen_values_ptr[3*tid+2] = matrix_s[2];
    }
  }

  __global__ void
  computeOmegaCr(const float * dev_eigen_values_ptr, std::size_t num_points, float * dev_omega_cr)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < num_points)
    {
      float lambda_0 = dev_eigen_values_ptr[3*tid];
      float lambda_1 = dev_eigen_values_ptr[3*tid+1];
      float lambda_2 = dev_eigen_values_ptr[3*tid+2];
      dev_omega_cr[tid] = fmaxf (lambda_1-lambda_0, fabsf (lambda_2-(lambda_1+lambda_0))) / lambda_2;
    }
  }

  __global__ void
  computeOmegaB_2(const float * dev_eigen_values_ptr, std::size_t num_points, float * dev_omega_b_2)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < num_points)
    {
    }
    // // Calculate penalty function for detecting board points.
    // // Calculate beta.
    // std::vector<Eigen::Vector3f> project_vectors(neighbor_indices.size () - 1);
    // for (std::size_t idx_neighbor = 1; idx_neighbor < neighbor_indices.size(); ++idx_neighbor)
    // {
    //   const pcl::PointXYZRGB & point_neighbor = (*point_cloud_)[neighbor_indices[idx_neighbor]];
    //   Eigen::Vector3f point_q (point_neighbor.x, point_neighbor.y, point_neighbor.z);
    //   Eigen::Vector3f vect_pq = point_q - point_p;
    //   project_vectors[idx_neighbor-1] = vect_pq - vect_pq.dot(eig_vector_0)/eig_vector_0.dot(eig_vector_0) * eig_vector_0;
    // }

    // float beta = calMaxIncludedAngle(project_vectors, eig_vector_0);
    // omega_b_2_list[idx_point] = 1.0 - beta / (2.0*3.14151926);

  }

  __device__ void bidiagonalize(const float * matrix_a, int num_dim, float * matrix_u, float * matrix_b, float * matrix_v);
  __device__ void svd_bidiagonal_matrix(float * matrix_b, float * matrix_u, float * matrix_v, int num_dim, float tolerance);
  __device__ void golub_kahan_svd_step(float * matrix_b, float * matrix_p, float * matrix_q, int num_dim);
  
  // SRL_EPS = 1E-7;
  // SRL_PI = 3.1415927;
  __constant__ float SRL_EPS = 1E-7;
  __constant__ float SRL_PI = 3.1415927;

  __device__ float srl_sign(float value)
  {
      if (value >= 0.0)
          return (1.0);
      else
          return (-1.0);
  }
  
  __device__ float srl_sqrt(float scalar)
  {
      if ((scalar > -SRL_EPS) && (scalar < 0.0))
          scalar = 0.0;
  
      return (sqrtf(scalar));
  }
  
  __device__ float srl_asin(float scalar)
  {
      if ((scalar > 1.0) && (scalar < (1.0+SRL_EPS)))
          scalar = 1.0;
      else if ((scalar < -1.0) && (scalar > (-1.0-SRL_EPS)))
          scalar = -1.0;
  
      return (asinf(scalar));
  }
  
  __device__ float srl_acos(float scalar)
  {
      if ((scalar > 1.0) && (scalar < (1.0+SRL_EPS)))
          scalar = 1.0;
      else if ((scalar < -1.0) && (scalar > (-1.0-SRL_EPS)))
          scalar = -1.0;
  
      return (acosf(scalar));
  }
  
  __device__ void srl_svd (const float * matrix_a, int num_dim, float * matrix_u, float * matrix_s, float * matrix_v, float tolerance)
  {
      float * mat_u = (float *)malloc((num_dim*num_dim)*sizeof(float));
      float * mat_v = (float *)malloc((num_dim*num_dim)*sizeof(float));
      float * mat_b = (float *)malloc((num_dim*num_dim)*sizeof(float));
      bidiagonalize(matrix_a, num_dim, mat_u, mat_b, mat_v);
  
      float * mat_p = (float *)malloc((num_dim*num_dim)*sizeof(float));
      float * mat_q = (float *)malloc((num_dim*num_dim)*sizeof(float));
      svd_bidiagonal_matrix(mat_b, mat_p, mat_q, num_dim, tolerance);
  
      srl_matrix_multiply(mat_u, mat_p, num_dim, num_dim, num_dim, matrix_u);
      srl_matrix_multiply(mat_v, mat_q, num_dim, num_dim, num_dim, matrix_v);
      for (int i = 0; i < (num_dim*num_dim); ++i)
      {
          matrix_s[i] = mat_b[i];
      }
  
      free(mat_u);
      free(mat_v);
      free(mat_b);
      free(mat_p);
      free(mat_q);
  }
  
  __device__ void bidiagonalize(const float * matrix_a, int num_dim, float * matrix_u, float * matrix_b, float * matrix_v)
  {
      srl_matrix_copy(matrix_a, num_dim*num_dim, matrix_b);
      srl_matrix_identity(matrix_u, num_dim, num_dim);
      srl_matrix_identity(matrix_v, num_dim, num_dim);
  
      for (std::size_t k = 0; k < num_dim; ++k)
      {
          float * vect_householder = (float *)malloc(num_dim*sizeof(float));
          for (std::size_t i = 0; i < num_dim; ++i)
              vect_householder[i] = matrix_b[i+k*num_dim];
  
          for (std::size_t i = 0; i < k; ++i)
          {
              vect_householder[i] = 0.0;
          }
          vect_householder[k] = 1.0;
  
          float sigma = 0.0;
          if (k+1 < num_dim)
          {
              for (int i = k+1; i < num_dim; ++i)
              {
                  sigma += matrix_b[i+k*num_dim] * matrix_b[i+k*num_dim];
              }
          }
  
          float beta;
          if (sigma < SRL_EPS && matrix_b[k+k*num_dim] >= 0)
          {
              beta = 0.0;
          }
          else if (sigma < SRL_EPS && matrix_b[k+k*num_dim] < 0)
          {
              beta = -2.0;
          }
          else
          {
              float mu = srl_sqrt(powf(matrix_b[k+k*num_dim],2) + sigma);
              float k_elem;
              if (matrix_b[k+k*num_dim] < 0)
              {
                  k_elem = matrix_b[k+k*num_dim] - mu;
              }
              else
              {
                  k_elem = -sigma / (matrix_b[k+k*num_dim] + mu);
              }
  
              beta = 2.0 * powf(k_elem, 2) / (sigma + powf(k_elem,2));
              vect_householder[k] = k_elem;
              for (int i = 0; i < num_dim; ++i)
                  vect_householder[i] /= k_elem;
          }
  
          // Householder reflection matrix.
          float * mat_q = (float *)malloc((num_dim*num_dim)*sizeof(float));
          for (int i = 0; i < num_dim; ++i)
          {
              for (int j = 0; j < num_dim; ++j)
              {
                  if (i == j)
                      mat_q[i+j*num_dim] = 1.0 - beta*vect_householder[i]*vect_householder[j];
                  else
                      mat_q[i+j*num_dim] = 0.0 - beta*vect_householder[i]*vect_householder[j];
              }
          }
  
          float * matrix_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
          srl_matrix_multiply(mat_q, matrix_b, num_dim, num_dim, num_dim, matrix_temp);
          srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);
  
          srl_matrix_multiply(matrix_u, mat_q, num_dim, num_dim, num_dim, matrix_temp);
          srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_u);
  
          free(matrix_temp);
          free(vect_householder);
  
          if (k < num_dim-2)
          {
              float * vect_householder = (float *)malloc(num_dim*sizeof(float));
              for (std::size_t j = 0; j < num_dim; ++j)
                  vect_householder[j] = matrix_b[k+j*num_dim];
  
              for (std::size_t i = 0; i < k+1; ++i)
              {
                  vect_householder[i] = 0.0;
              }
              vect_householder[k+1] = 1.0;
  
              float sigma = 0.0;
              for (int i = k+2; i < num_dim; ++i)
                  sigma += matrix_b[k+i*num_dim] * matrix_b[k+i*num_dim];
  
              float beta;
              if (sigma < SRL_EPS && matrix_b[k+(k+1)*num_dim] >= 0)
              {
                  beta = 0.0;
              }
              else if (sigma < SRL_EPS && matrix_b[k+(k+1)*num_dim] < 0)
              {
                  beta = -2.0;
              }
              else
              {
                  float mu = srl_sqrt(powf(matrix_b[k+(k+1)*num_dim], 2) + sigma);
                  float temp;   // (k+1)-th element;
                  if (matrix_b[k+(k+1)*num_dim] < 0.0)
                  {
                      temp = matrix_b[k+(k+1)*num_dim] - mu;
                  }
                  else
                  {
                      temp = -sigma / (matrix_b[k+(k+1)*num_dim] + mu);
                  }
                  beta = 2.0 * powf(temp, 2) / (sigma + powf(temp, 2));
                  vect_householder[k+1] = temp;
                  for (int i = 0; i < num_dim; ++i)
                      vect_householder[i] /= temp;
              }
  
              // Householder reflection matrix.
              float * mat_p = (float *)malloc((num_dim*num_dim)*sizeof(float));
              for (int i = 0; i < num_dim; ++i)
              {
                  for (int j = 0; j < num_dim; ++j)
                  {
                      if (i == j)
                          mat_p[i+j*num_dim] = 1.0 - beta*vect_householder[i]*vect_householder[j];
                      else
                          mat_p[i+j*num_dim] = 0.0 - beta*vect_householder[i]*vect_householder[j];
                  }
              }
  
              // matrixB = matrixB * matP;
              // matrixV = matrixV * matP;
              float * matrix_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
              srl_matrix_multiply(matrix_b, mat_p, num_dim, num_dim, num_dim, matrix_temp);
              srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);
  
              srl_matrix_multiply(matrix_v, mat_p, num_dim, num_dim, num_dim, matrix_temp);
              srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_v);
  
              free(matrix_temp);
              free(vect_householder);
          }
      }
  }
  
  __device__ void svd_bidiagonal_matrix(float * matrix_b, float * matrix_u, float * matrix_v, int num_dim, float tolerance)
  {
      srl_matrix_identity(matrix_u, num_dim, num_dim);
      srl_matrix_identity(matrix_v, num_dim, num_dim);
  
      while (true)
      {
          for (std::size_t i = 0; i < num_dim-1; ++i)
          {
              // if (fabsf(matrix_b[i+(i+1)*num_dim]) <= tolerance*(fabsf(matrix_b[i+i*num_dim])+fabsf(matrix_b[i+1+(i+1)*num_dim])))
              if (fabsf(matrix_b[i+(i+1)*num_dim]) <= SRL_EPS)
              {
                  matrix_b[i+(i+1)*num_dim] = 0.0;
              }
          }
  
          // Find the largest q and the smallest p.
          std::size_t q = 0;
          for (int i = num_dim-1; i >= 0; --i)
          {
              if (0 == i)
              {
                  q++;
                  break;
              }
  
              if (fabsf(matrix_b[(i-1)+i*num_dim]) < SRL_EPS)
              {
                  q++;
              }
              else
              {
                  break;
              }
          }
  
          if (q == num_dim)
          {
              break;
          }
  
          std::size_t dimB_2 = 0;
          for (int i = num_dim-q-1; i >=0 ; --i)
          {
              if (0 == i) {
                  dimB_2++;
                  break;
              }
  
              if (fabsf(matrix_b[(i-1)+i*num_dim]) > SRL_EPS)
              {
                  dimB_2++;
              }
              else
              {
                  dimB_2++;
                  break;
              }
          }
  
          std::size_t p = num_dim - q - dimB_2;
  
          float * matB_2 = (float *)malloc((dimB_2*dimB_2)*sizeof(float));
          for (int i = 0; i < dimB_2; ++i)
              for (int j = 0; j < dimB_2; ++j)
                  matB_2[i+j*dimB_2] = matrix_b[(p+i)+(p+j)*num_dim];
  
          bool flag_zero_diag = false;
          for (std::size_t i = 0; i < dimB_2-1; ++i)
          {
              if (fabsf(matB_2[i+i*dimB_2]) < SRL_EPS)
              {
                  flag_zero_diag = true;
                  // Zero the superdiagonal entry in the same row.
                  // Construct Givens rotation matrix.
                  float * givens_matrix = (float *)malloc((num_dim*num_dim)*sizeof(float));
                  srl_matrix_identity(givens_matrix, num_dim, num_dim);
  
                  float c = 0.0;
                  float s = -srl_sign(matB_2[i+(i+1)*dimB_2]);
                  givens_matrix[(p+i)+(p+i)*num_dim] = c;
                  givens_matrix[(p+i+1)+(p+i+1)*num_dim] = c;
                  givens_matrix[(p+i)+(p+i+1)*num_dim] = s;
                  givens_matrix[(p+i+1)+(p+i)*num_dim] = -s;
  
                  // matrixB *= givensMatrix;
                  // matrixV *= givensMatrix;
                  float * matrix_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
                  srl_matrix_multiply(matrix_b, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
                  srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);
  
                  srl_matrix_multiply(matrix_v, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
                  srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_v);
  
                  free(matrix_temp);
                  free(givens_matrix);
              }
          }
  
          if (!flag_zero_diag)
          {
              // GolubKahanSVDStep(matB_2, matP, matQ);
              float * mat_p = (float *)malloc((dimB_2*dimB_2)*sizeof(float));
              float * mat_q = (float *)malloc((dimB_2*dimB_2)*sizeof(float));
              golub_kahan_svd_step(matB_2, mat_p, mat_q, dimB_2);
  
              for (int i = 0; i < dimB_2; ++i)
                  for (int j = 0; j < dimB_2; ++j)
                      matrix_b[(p+i)+(p+j)*num_dim] = matB_2[i+j*dimB_2];
  
              float * mat_p_full = (float *)malloc((num_dim*num_dim)*sizeof(float));
              float * mat_q_full = (float *)malloc((num_dim*num_dim)*sizeof(float));
              srl_matrix_identity(mat_p_full, num_dim, num_dim);
              srl_matrix_identity(mat_q_full, num_dim, num_dim);
  
              for (int m = 0; m < dimB_2; ++m)
              {
                  for (int n = 0; n < dimB_2; ++n)
                  {
                      mat_p_full[(p+m)+(p+n)*num_dim] = mat_p[m+n*dimB_2];
                      mat_q_full[(p+m)+(p+n)*num_dim] = mat_q[m+n*dimB_2];
                  }
              }
  
              float * mat_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
              srl_matrix_multiply(matrix_u, mat_p_full, num_dim, num_dim, num_dim, mat_temp);
              srl_matrix_copy(mat_temp, num_dim*num_dim, matrix_u);
  
              srl_matrix_multiply(matrix_v, mat_q_full, num_dim, num_dim, num_dim, mat_temp);
              srl_matrix_copy(mat_temp, num_dim*num_dim, matrix_v);
  
              free(mat_temp);
              free(mat_p_full);
              free(mat_q_full);
              free(mat_p);
              free(mat_q);
          }
  
          free(matB_2);
      }
  }
  
  __device__ void golub_kahan_svd_step(float * matrix_b, float * matrix_p, float * matrix_q, int num_dim)
  {
      srl_matrix_identity(matrix_p, num_dim, num_dim);
      srl_matrix_identity(matrix_q, num_dim, num_dim);
  
      float * matrix_b_t = (float *)malloc((num_dim*num_dim)*sizeof(float));
      srl_matrix_transpose(matrix_b, num_dim, num_dim, matrix_b_t);
      float * mat_t = (float *)malloc((num_dim*num_dim)*sizeof(float));
      srl_matrix_multiply(matrix_b_t, matrix_b, num_dim, num_dim, num_dim, mat_t);
  
      float mat_c[4];
      mat_c[0] = mat_t[(num_dim-2)+(num_dim-2)*num_dim];
      mat_c[1] = mat_t[(num_dim-2+1)+(num_dim-2)*num_dim];
      mat_c[2] = mat_t[(num_dim-2)+(num_dim-2+1)*num_dim];
      mat_c[3] = mat_t[(num_dim-2+1)+(num_dim-2+1)*num_dim];
      float coeff_b = -(mat_c[0] + mat_c[3]);
      float coeff_c = mat_c[0]*mat_c[3] - mat_c[2]*mat_c[1];
      float eig_value_mat_c_1 = (-coeff_b + srl_sqrt(powf(coeff_b, 2)-4.0*coeff_c)) / 2.0;
      float eig_value_mat_c_2 = (-coeff_b - srl_sqrt(powf(coeff_b, 2)-4.0*coeff_c)) / 2.0;
      float mu = fabsf(eig_value_mat_c_1-mat_c[3]) < fabsf(eig_value_mat_c_2-mat_c[3]) ? eig_value_mat_c_1 : eig_value_mat_c_2;
  
      float alpha = mat_t[0] - mu;
      float beta = mat_t[num_dim];
      for (std::size_t k = 0; k < num_dim-1; ++k)
      {
          float c, s;
          if (fabsf(beta) < SRL_EPS)
          {
              c = srl_sign(alpha);
              s = 0.0;
          }
          else
          {
              s = -beta / srl_sqrt(powf(alpha, 2)+powf(beta, 2));
              c = alpha / srl_sqrt(powf(alpha, 2)+powf(beta, 2));
          }
  
          float * givens_matrix = (float *)malloc((num_dim*num_dim)*sizeof(float));
          srl_matrix_identity(givens_matrix, num_dim, num_dim);
          givens_matrix[k+k*num_dim] = c;
          givens_matrix[(k+1)+(k+1)*num_dim] = c;
          givens_matrix[k+(k+1)*num_dim] = s;
          givens_matrix[(k+1)+k*num_dim] = -s;
  
          // matrixB *= givensMatrix;
          // matrixQ *= givensMatrix;
          float * matrix_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
          srl_matrix_multiply(matrix_b, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
          srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);
  
          srl_matrix_multiply(matrix_q, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
          srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_q);
  
          alpha = matrix_b[k+k*num_dim];
          beta = matrix_b[(k+1)+k*num_dim];
          if (fabsf(beta) < SRL_EPS)
          {
              c = srl_sign(alpha);
              s = 0.0;
          }
          else
          {
              s = -beta / srl_sqrt(powf(alpha,2)+powf(beta,2));
              c = alpha / srl_sqrt(powf(alpha,2)+powf(beta,2));
          }
  
          givens_matrix[k+k*num_dim] = c;
          givens_matrix[(k+1)+(k+1)*num_dim] = c;
          givens_matrix[k+(k+1)*num_dim] = s;
          givens_matrix[(k+1)+k*num_dim] = -s;
  
          float * givens_matrix_t = (float *)malloc((num_dim*num_dim)*sizeof(float));
          srl_matrix_transpose(givens_matrix, num_dim, num_dim, givens_matrix_t);
          srl_matrix_multiply(givens_matrix_t, matrix_b, num_dim, num_dim, num_dim, matrix_temp);
          srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);
  
          srl_matrix_multiply(matrix_p, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
          srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_p);
  
          free(givens_matrix);
          free(givens_matrix_t);
          free(matrix_temp);
  
          if (k < num_dim-2)
          {
              alpha = matrix_b[k+(k+1)*num_dim];
              beta = matrix_b[k+(k+2)*num_dim];
          }
      }
  
      free(matrix_b_t);
      free(mat_t);
  }
  
  __device__ void srl_matrix_identity (float * mat_a, int num_row, int num_column)
  {
      for (int i = 0; i < num_row; ++i)
      {
          for (int j = 0; j < num_column; ++j)
          {
              if (i == j)
                  mat_a[i+j*num_row] = 1.0;
              else
                  mat_a[i+j*num_row] = 0.0;
          }
      }
  }
  
  __device__ void srl_matrix_zeros (float * mat_a, int num_row, int num_column)
  {
      for (int i = 0; i < (num_row*num_column); ++i)
          mat_a[i] = 0.0;
  }
  
  __device__ void srl_matrix_ones (float * mat_a, int num_row, int num_column)
  {
      for (int i = 0; i < (num_row*num_column); ++i)
          mat_a[i] = 1.0;
  }
  
  __device__ void srl_matrix_copy (const float * mat_src, int length , float * mat_dst)
  {
      for (int i = 0; i < length; ++i)
          mat_dst[i] = mat_src[i];
  }
  
  __device__ void srl_matrix_add (const float * mat_a, const float * mat_b, int num_row, int num_column, float * mat_c)
  {
      for (int i = 0; i < num_row; ++i)
          for (int j = 0; j < num_column; ++j)
              mat_c[i+j*num_row] = mat_a[i+j*num_row] + mat_b[i+j*num_row];
  }
  
  __device__ void srl_matrix_subtract (const float * mat_a, const float * mat_b, int num_row, int num_column, float * mat_c)
  {
      for (int i = 0; i < num_row; ++i)
          for (int j = 0; j < num_column; ++j)
              mat_c[i+j*num_row] = mat_a[i+j*num_row] - mat_b[i+j*num_row];
  }
  
  __device__ void srl_matrix_multiply (const float * mat_a, const float * mat_b, int num_row_a, int num_mid, int num_col_b, float * mat_c)
  {
      for (int i = 0; i < num_row_a; ++i)
      {
          for (int j = 0; j < num_col_b; ++j)
          {
              mat_c[i+j*num_row_a] = 0.0;
              for (int k = 0; k < num_mid; ++k)
              {
                  mat_c[i+j*num_row_a] += mat_a[i+k*num_row_a] * mat_b[k+j*num_mid];
              }
          }
      }
  }
  
  __device__ void srl_matrix_multiplied_by_scalar (const float * mat_a, int num_row, int num_column, float scalar, float * mat_b)
  {
      for (int i = 0; i < num_row; ++i)
          for (int j = 0; j < num_column; ++j)
              mat_b[i+j*num_row] = mat_a[i+j*num_row] * scalar;
  }
  
  __device__ void srl_matrix_divided_by_scalar (const float * mat_a, int num_row, int num_column, float scalar, float * mat_b)
  {
      for (int i = 0; i < num_row; ++i)
          for (int j = 0; j < num_column; ++j)
              mat_b[i+j*num_row] = mat_a[i+j*num_row] / scalar;
  }
  
  __device__ void srl_matrix_transpose (const float * mat_a, int num_row, int num_column, float * mat_t)
  {
      for (int i = 0; i < num_column; ++i)
      {
          for (int j = 0; j < num_row; ++j)
          {
              mat_t[i+j*num_column] = mat_a[j+i*num_row];
          }
      }
  }
  
  __device__ float srl_dot_product (const float * mat_a, const float * mat_b, int length)
  {
      float result = 0.0;
      for (int i = 0; i < length; ++i)
          result += mat_a[i]*mat_b[i];
  
      return (result);
  }



} // namespace radi
