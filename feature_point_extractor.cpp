#include <numeric>
#include <boost/range/numeric.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/make_shared.hpp>
#include <Eigen/Eigenvalues>

#include <pcl/octree/octree_search.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/filters/extract_indices.h>

// Test
#include <pcl/visualization/pcl_visualizer.h>

#include "feature_point_extractor.h"

namespace radi
{

  float
  calMaxIncludedAngle(const std::vector<Eigen::Vector3f> & vectors, const Eigen::Vector3f & normal);

  FeaturePointExtractor::FeaturePointExtractor() : k_(30), corner_point_indices_(std::vector<int>()),
      edge_point_indices_(std::vector<int>()), board_point_indices_(std::vector<int>())
  { }

  FeaturePointExtractor::~FeaturePointExtractor()
  { }

  void
  FeaturePointExtractor::setInputCloud (const PointCloudConstPtr & point_cloud)
  {
    point_cloud_ = point_cloud;
  }

  void
  FeaturePointExtractor::compute ()
  {
    std::vector<Eigen::Vector3f> omega_cr_list (point_cloud_->size ());
    std::vector<float> omega_b_2_list (point_cloud_->size ());
    std::vector<std::vector<int> > neighbor_indices_list (point_cloud_->size ());

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree (0.001);
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
      octree.radiusSearch(point, 0.02, neighbor_indices, neighbor_distances, 500);
      neighbor_indices_list[idx_point] = neighbor_indices;

      std::cout << "Size of neighborhood: " << neighbor_indices.size() << std::endl;


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
    float tau = 0.5;
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

} // namespace radi
