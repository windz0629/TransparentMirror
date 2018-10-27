#include <cmath>
#include <random>
#include <exception>

#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "icp.h"
#include "distance_measurer.h"

namespace radi
{
  const float PI = 3.1415926;

  IterativeClosestPoint::IterativeClosestPoint ()
  {
    translation_deviation_ = Eigen::Vector3f (0.05, 0.05, 0.05);
    rotation_deviation_ = Eigen::Vector3f (PI/6.0, PI/6.0, PI/6.0);
    iteration_outer_ = 100;
    iteration_inner_ = 30;
    threshold_distance_ = 0.01;
    init_transf_ = Eigen::MatrixXf::Identity (4,4);
  }

  IterativeClosestPoint::IterativeClosestPoint (const PointCloudConstPtr & refer_point_cloud,
      const PointCloudConstPtr & scene_point_cloud)
  {
    refer_point_cloud_ = refer_point_cloud;
    scene_point_cloud_ = scene_point_cloud;
  }

  IterativeClosestPoint::~IterativeClosestPoint ()
  { }

  void
  IterativeClosestPoint::setReferCloud (const PointCloudConstPtr & refer_point_cloud)
  {
    refer_point_cloud_ = refer_point_cloud;
  }

  void
  IterativeClosestPoint::setSceneCloud (const PointCloudConstPtr & scene_point_cloud)
  {
    scene_point_cloud_ = scene_point_cloud;
    point_cloud_used_ = scene_point_cloud_;
  }

  void
  IterativeClosestPoint::setIndices (const IndicesConstPtr & indices)
  {
    if (indices->empty())
    {
      indices_ = NULL;
      point_cloud_used_ = scene_point_cloud_;
    }
    else
    {
      indices_ = indices;
      IterativeClosestPoint::PointCloud::Ptr point_cloud (new IterativeClosestPoint::PointCloud ());
      pcl::ExtractIndices<pcl::PointXYZRGB> extractor;
      extractor.setInputCloud (scene_point_cloud_);
      extractor.setIndices (indices_);
      extractor.filter (*point_cloud);

      point_cloud_used_ = point_cloud;
    }
  }

  void
  IterativeClosestPoint::setDeviationTranslation (const Eigen::Vector3f & translation_deviation)
  {
    translation_deviation_ = translation_deviation;
  }

  void
  IterativeClosestPoint::setDeviationRotation (const Eigen::Vector3f & rotation_deviation)
  {
    rotation_deviation_ = rotation_deviation;
  }

  void
  IterativeClosestPoint::setIterationOuter (int iteration_outer)
  {
    iteration_outer_ = iteration_outer;
  }

  void
  IterativeClosestPoint::setIterationInner (int iteration_inner)
  {
    iteration_inner_ = iteration_inner;
  }

  void
  IterativeClosestPoint::setInitialTransformation (const Eigen::Matrix4f & init_transf)
  {
    init_transf_ = init_transf;
  }

  void
  IterativeClosestPoint::setThresholdDistance (float threshold_distance)
  {
    threshold_distance_ = threshold_distance;
  }

  void
  IterativeClosestPoint::setThresholdAverageDistance (float threshold_average_distance)
  {
    threshold_average_distance_ = threshold_average_distance;
  }

  IterativeClosestPoint::PointCloudConstPtr
  IterativeClosestPoint::getReferPointCloud () const
  {
    return (refer_point_cloud_);
  }

  IterativeClosestPoint::PointCloudConstPtr
  IterativeClosestPoint::getScenePointCloud () const
  {
    return (scene_point_cloud_);
  }

  void
  IterativeClosestPoint::getDeviationTranslation (Eigen::Vector3f & translation_deviation)
  {
    translation_deviation = translation_deviation_;
  }

  void
  IterativeClosestPoint::getDeviationRotation (Eigen::Vector3f & rotation_deviation)
  {
    rotation_deviation = rotation_deviation_;
  }

  int
  IterativeClosestPoint::getIterationOuter ()
  {
    return (iteration_outer_);
  }

  int
  IterativeClosestPoint::getIterationInner ()
  {
    return (iteration_inner_);
  }

  const Eigen::Matrix4f
  IterativeClosestPoint::getInitialTransformation ()
  {
    return (init_transf_);
  }

  void
  IterativeClosestPoint::getThresholdDistance (float & threshold_distance)
  {
    threshold_distance = threshold_distance_;
  }

  void
  IterativeClosestPoint::getThresholdAverageDistance (float & threshold_average_distance)
  {
    threshold_average_distance = threshold_average_distance_;
  }

  float
  IterativeClosestPoint::calObjectiveValue (const Eigen::Matrix4f & mat_transf)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_scene (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::transformPointCloud(*point_cloud_used_, *transformed_scene, mat_transf);

    radi::DistanceMeasurer dist_measurer;
    dist_measurer.setReferPointCloud(refer_point_cloud_);

    float objective_value = 0.0;
    for (std::size_t i = 0; i < (*transformed_scene).points.size(); ++i)
    {
      float shortest_distance = dist_measurer.calShortestDistance((*transformed_scene).points[i]);
      if (shortest_distance > threshold_distance_)
      {
        throw "Too large distance.";
      }

      if (shortest_distance > distance_limit_)
        objective_value += 20*shortest_distance;
      else
        objective_value += shortest_distance;
    }

    return objective_value;
  }

  bool
  IterativeClosestPoint::estimate (Eigen::Matrix4f & estimated_transf)
  {
    Eigen::Matrix4f mat_transf = init_transf_;
    float objective_value;
    try
    {
      objective_value = calObjectiveValue(mat_transf);
    }
    catch (char const * msg)
    {
      objective_value = 99.0 * static_cast<float> (point_cloud_used_->size ());
    }

    for (std::size_t idx_outer = 0; idx_outer < iteration_outer_; ++idx_outer)
    {
      Eigen::Vector3f translation = mat_transf.block (0,3,3,1);
      Eigen::Vector3f rotation = matrix2euler (mat_transf.block (0,0,3,3));

      Eigen::Vector3f translation_sampled = uniformRandom (translation-translation_deviation_,
              translation+translation_deviation_);

      std::vector<float> objective_value_list (iteration_inner_);
      std::vector<Eigen::Matrix4f> mat_transf_list (iteration_inner_);
      for (std::size_t idx_inner = 0; idx_inner < iteration_inner_; ++idx_inner)
      {
        Eigen::Vector3f rotation_sampled = uniformRandom (rotation-rotation_deviation_,
                rotation+rotation_deviation_);
        mat_transf_list[idx_inner] = Eigen::Matrix4Xf::Identity (4,4);
        mat_transf_list[idx_inner].block (0,0,3,3) = euler2matrix (rotation_sampled);
        mat_transf_list[idx_inner].block (0,3,3,1) = translation_sampled;

        try
        {
          objective_value_list[idx_inner] = calObjectiveValue (mat_transf_list[idx_inner]);
        }
        catch (char const * msg)
        {
          objective_value_list[idx_inner] = 99.0 * static_cast<float> (point_cloud_used_->size ());
        }
      }

      // Sort 'objective_value' and select the best sample.
      std::vector<std::size_t> order_indices (iteration_inner_);
      std::iota(order_indices.begin (), order_indices.end (), 0);
      std::sort(order_indices.begin (), order_indices.end (), [&objective_value_list](int idx_1, int idx_2)
              { return objective_value_list[idx_1] <= objective_value_list[idx_2]; });

      if (objective_value_list[order_indices[0]] > objective_value)
      {
        objective_value = objective_value_list[order_indices[0]];
        mat_transf = mat_transf_list[order_indices[0]];
      }
    }

    if (objective_value < (static_cast<float> (point_cloud_used_->size ()) * threshold_average_distance_))
    {
      estimated_transf = mat_transf;
      return (true);
    }
    else
    {
      return (false);
    }
  }

  const Eigen::Vector3f
  uniformRandom (const Eigen::Vector3f & min_boundary, const Eigen::Vector3f & max_boundary)
  {
      Eigen::Vector3f random_value;
      std::random_device rand_device;
      std::mt19937 generator (rand_device());
      std::uniform_real_distribution<float> distr_x (min_boundary[0], max_boundary[0]);
      std::uniform_real_distribution<float> distr_y (min_boundary[1], max_boundary[1]);
      std::uniform_real_distribution<float> distr_z (min_boundary[2], max_boundary[2]);
      random_value[0] = distr_x (generator);
      random_value[1] = distr_y (generator);
      random_value[2] = distr_z (generator);

      return (random_value);
  }

  const Eigen::Vector3f
  gaussianRandom(const Eigen::Vector3f & mean, const Eigen::Vector3f & deviation)
  {
      Eigen::Vector3f random_value;
      std::random_device rand_device;
      std::mt19937 generator (rand_device());
      std::normal_distribution<float> distr_x (mean[0], deviation[0]);
      std::normal_distribution<float> distr_y (mean[1], deviation[1]);
      std::normal_distribution<float> distr_z (mean[2], deviation[2]);
      random_value[0] = distr_x (generator);
      random_value[1] = distr_y (generator);
      random_value[2] = distr_z (generator);

      return (random_value);
  }

  const Eigen::Vector3f
  gaussianRandom(const Eigen::Vector3f & mean, float deviation)
  {
      Eigen::Vector3f std_deviation;
      std_deviation[0] = deviation;
      std_deviation[1] = deviation;
      std_deviation[2] = deviation;
      return (gaussianRandom (mean, std_deviation));
  }

  const Eigen::Vector3f
  matrix2euler (const Eigen::Matrix3f & mat_rotation)
  {
    const float EPS = 1.0E-8;

    float alpha;
    float beta;
    float gamma;

    // Assume beta is in [0,pi].
    double a_02 = mat_rotation (0,2);
    double a_01 = mat_rotation (0,1);
    double a_11 = mat_rotation (1,1);
    double a_12 = mat_rotation (1,2);
    double a_20 = mat_rotation (2,0);
    double a_21 = mat_rotation (2,1);
    double a_22 = mat_rotation (2,2);

    beta = std::atan2 (std::sqrt (std::pow (a_02,2)+std::pow (a_12,2)), a_22);

    if ((EPS < beta) && (beta < (PI-EPS)))
    {
      alpha = std::atan2 (a_12, a_02);
      gamma = std::atan2 (a_21, -a_20);
    }
    else if (beta <= EPS)
    {
      alpha = 0.0;
      gamma = std::atan2 (-a_01, a_11);
    }
    else
    {
      alpha = 0.0;
      gamma = std::atan2 (a_01, a_11);
    }

    return (Eigen::Vector3f (alpha, beta, gamma));
  }

  const Eigen::Matrix3f
  euler2matrix (const Eigen::Vector3f & euler_angle)
  {
    double phi = euler_angle[0];
    double theta = euler_angle[1];
    double psi = euler_angle[2];

    Eigen::Matrix3f mat_rotation;
    mat_rotation (0,0) = std::cos (phi)*std::cos (theta)*std::cos (psi) - std::sin (phi)*std::sin (psi);
    mat_rotation (0,1) = -std::cos (phi)*std::cos (theta)*std::sin (psi) - std::sin (phi)*std::cos (psi);
    mat_rotation (0,2) = std::cos (phi)*std::sin (theta);

    mat_rotation (1,0) = std::sin (phi)*std::cos (theta)*std::cos (psi) + std::cos (phi)*std::sin (psi);
    mat_rotation (1,1) = -std::sin (phi)*std::cos (theta)*std::sin (psi) + std::cos (phi)*std::cos (psi);
    mat_rotation (1,2) = std::sin (phi)*std::sin (theta);

    mat_rotation (2,0) = -std::sin (theta)*std::cos (psi);
    mat_rotation (2,1) = std::sin (theta)*std::sin (psi);
    mat_rotation (2,2) = std::cos (theta);

    return (mat_rotation);
  }

} // namespace radi
