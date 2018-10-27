#include <cmath>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "lgf_corresp_group.h"
#include "icp.h"

namespace radi
{

  const float PI = 3.1415926;

  bool
  pairFeatures (const CornerEdgesFeature & scene_feature, const CornerEdgesFeature & refer_feature,
      std::vector<std::vector<pcl::Correspondence> > & angle_corresps_list);

  bool
  pairFeatures (const CircleNormalFeature & scene_feature, const CircleNormalFeature & refer_feature,
      std::vector<std::vector<pcl::Correspondence> > & angle_corresps_list);

  LGFCorrespGroup::LGFCorrespGroup () : angle_threshold_(0.3), distance_threshold_(0.03),
      radius_threshold_(0.03), resolution_(0.5), filtered_point_cloud_(new pcl::PointCloud<pcl::PointXYZRGB> ()),
      transform_candidates_(std::vector<Eigen::Matrix4f>()), objective_value_(1E10), rate_threshold_(0.3),
      rough_transform_(Eigen::MatrixXf::Identity(4,4))
  { }

  LGFCorrespGroup::~LGFCorrespGroup ()
  { }

  void
  LGFCorrespGroup::setReferenceCloud (const PointCloudConstPtr & reference_point_cloud)
  {
    reference_point_cloud_ = reference_point_cloud;
  }

  void
  LGFCorrespGroup::setSceneCloud (const PointCloudConstPtr & scene_point_cloud)
  {
    scene_point_cloud_ = scene_point_cloud;
  }

  void
  LGFCorrespGroup::setFeatureIndices (const IndicesConstPtr & feature_point_indices)
  {
    feature_point_indices_ = feature_point_indices;
  }

  void
  LGFCorrespGroup::setReferenceFeatures (const std::vector<CornerEdgesFeature> * reference_corner_features,
      const std::vector<CircleNormalFeature> * reference_circle_features)
  {
    reference_corner_features_ = reference_corner_features;
    reference_circle_features_ = reference_circle_features;
  }

  void
  LGFCorrespGroup::setSceneFeatures (const std::vector<CornerEdgesFeature> * scene_corner_features,
      const std::vector<CircleNormalFeature> * scene_circle_features)
  {
    scene_corner_features_ = scene_corner_features;
    scene_circle_features_ = scene_circle_features;
  }

  bool
  LGFCorrespGroup::recognize (Eigen::Matrix4f & transf_matrix)
  {
    if (!feature_point_indices_->empty ())
    {
      pcl::ExtractIndices<pcl::PointXYZRGB> extractor;
      extractor.setInputCloud (scene_point_cloud_);
      extractor.setIndices (feature_point_indices_);
      extractor.filter (*filtered_point_cloud_);
    }
    else
    {
      pcl::copyPointCloud (*scene_point_cloud_, *filtered_point_cloud_);
    }

    this->transform_candidates_ = std::vector<Eigen::Matrix4f> ();

    if (!scene_corner_features_->empty ())
    {
      // Pair every feature in the scene to features in the reference.
      for (std::size_t idx_scene = 0; idx_scene < scene_corner_features_->size (); ++idx_scene)
      {
        for (std::size_t idx_refer = 0; idx_refer < reference_corner_features_->size (); ++idx_refer)
        {
          // vector pair candiates.
          const std::vector<float> & angle_seq_scene = (*scene_corner_features_)[idx_scene].getIncludedAngleSequence ();
          const std::vector<float> & angle_seq_refer = (*reference_corner_features_)[idx_refer].getIncludedAngleSequence ();
          std::vector<std::vector<pcl::Correspondence> > angle_corresps_list;
          if (pairFeatures(angle_seq_scene, angle_seq_refer, angle_corresps_list))
          {
            // Calcualte the transformation matrices.
            for (std::size_t idx_angle_corresps = 0; idx_angle_corresps < angle_corresps_list.size (); ++idx_angle_corresps)
            {
              Eigen::Matrix3f mat_covariance = Eigen::MatrixXf::Zero(3,3);
              for (std::size_t idx_angle = 0; idx_angle < angle_corresps_list[idx_angle_corresps].size (); ++idx_angle)
              {
                std::size_t idx_angle_scene = angle_corresps_list[idx_angle_corresps][idx_angle].index_query;
                std::size_t idx_angle_refer = angle_corresps_list[idx_angle_corresps][idx_angle].index_match;
                std::size_t idx_vector_scene = (*scene_corner_features_)[idx_scene].getEdgePairSequence()[idx_angle_scene][0];
                std::size_t idx_vector_refer = (*reference_corner_features_)[idx_refer].getEdgePairSequence()[idx_angle_refer][0];
                Eigen::Vector3f vect_scene = (*scene_corner_features_)[idx_scene].getEdge(idx_vector_scene);
                Eigen::Vector3f vect_refer = (*reference_corner_features_)[idx_refer].getEdge(idx_vector_refer);
                mat_covariance += vect_refer * vect_scene.transpose();
              }

              Eigen::JacobiSVD<Eigen::Matrix3f> svd_solver;
              svd_solver.compute(mat_covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

              Eigen::Matrix3f mat_rotation = svd_solver.matrixV() * svd_solver.matrixU().transpose();

              // Refer to: http://nghiaho.com/?page_id=671
              Eigen::Matrix4f mat_transf = Eigen::MatrixXf::Identity(4,4);
              mat_transf.block(0,0,3,3) = mat_rotation;
              mat_transf.block(0,3,3,1) = -mat_rotation * (*reference_corner_features_)[idx_refer].getCornerPosition ()
                      + (*scene_corner_features_)[idx_scene].getCornerPosition ();

              // Detect if the transformation has already been obtained.
              if (!hasObtained (mat_transf))
              {
                transform_candidates_.push_back (mat_transf);
              }
            }
          }
        }
      }
    }
    else if (!scene_circle_features_->empty ())
    {
      std::vector<float> objective_list;
      std::vector<Eigen::Matrix4f> transf_list;

      // Pair every feature in the scene to features in the reference.
      // 求解变换矩阵时，需要注意Normal方向可以反向。
      for (std::size_t idx_scene = 0; idx_scene < scene_circle_features_->size (); ++idx_scene)
      {
        for (std::size_t idx_refer = 0; idx_refer < reference_circle_features_->size (); ++idx_refer)
        {
          float radius_scene = (*scene_circle_features_)[idx_scene].getRadius ();
          float radius_refer = (*reference_circle_features_)[idx_refer].getRadius ();
          std::cout << "Radius in the scene: " << radius_scene << std::endl;
          std::cout << "Radius in the refer: " << radius_refer << std::endl;
          if (std::abs (radius_scene-radius_refer) < radius_threshold_)
          {
            if ((*scene_circle_features_)[idx_scene].getIncludedAngleSequence ().size () > 0)
            {
              const std::vector<float> & angle_seq_scene = (*scene_circle_features_)[idx_scene].getIncludedAngleSequence ();
              const std::vector<float> & angle_seq_refer = (*reference_circle_features_)[idx_refer].getIncludedAngleSequence ();
              std::vector<std::vector<pcl::Correspondence> > angle_corresps_list;
              if (pairFeatures(angle_seq_scene, angle_seq_refer, angle_corresps_list))
              {
                // Calcualte the transformation matrices.
                for (std::size_t idx_angle_corresps = 0; idx_angle_corresps < angle_corresps_list.size (); ++idx_angle_corresps)
                {
                  Eigen::Matrix3f mat_covariance = Eigen::MatrixXf::Zero(3,3);
                  for (std::size_t idx_angle = 0; idx_angle < angle_corresps_list[idx_angle_corresps].size (); ++idx_angle)
                  {
                    std::size_t idx_angle_scene = angle_corresps_list[idx_angle_corresps][idx_angle].index_query;
                    std::size_t idx_angle_refer = angle_corresps_list[idx_angle_corresps][idx_angle].index_match;
                    std::size_t idx_vector_scene = (*scene_circle_features_)[idx_scene].getPrincipalAxisPairSequence()[idx_angle_scene][0];
                    std::size_t idx_vector_refer = (*reference_circle_features_)[idx_refer].getPrincipalAxisPairSequence()[idx_angle_refer][0];
                    Eigen::Vector3f vect_scene = (*scene_circle_features_)[idx_scene].getPrincipalAxis(idx_vector_scene);
                    Eigen::Vector3f vect_refer = (*reference_circle_features_)[idx_refer].getPrincipalAxis(idx_vector_refer);
                    // mat_covariance += vect_scene * vect_model.transpose();
                    mat_covariance += vect_refer * vect_scene.transpose ();
                  }

                  // Backup convariant matrix.
                  Eigen::Matrix3f mat_covariance_temp = mat_covariance;

                  Eigen::Vector3f normal_scene = (*scene_circle_features_)[idx_scene].getNormal ();
                  Eigen::Vector3f normal_refer = (*reference_circle_features_)[idx_refer].getNormal ();
                  mat_covariance += normal_refer * normal_scene.transpose ();

                  Eigen::JacobiSVD<Eigen::Matrix3f> svd_solver;
                  svd_solver.compute(mat_covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

                  Eigen::Matrix3f mat_rotation = svd_solver.matrixV() * svd_solver.matrixU().transpose();

                  // Refer to: http://nghiaho.com/?page_id=671
                  Eigen::Matrix4f mat_transf = Eigen::MatrixXf::Identity(4,4);
                  mat_transf.block(0,0,3,3) = mat_rotation;
                  mat_transf.block(0,3,3,1) = -mat_rotation * (*reference_corner_features_)[idx_refer].getCornerPosition ()
                          + (*scene_corner_features_)[idx_scene].getCornerPosition ();

                  if (!hasObtained (mat_transf))
                  {
                    transform_candidates_.push_back (mat_transf);
                  }

                  mat_covariance = mat_covariance_temp + normal_refer * (-normal_scene).transpose ();

                  svd_solver.compute(mat_covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

                  mat_rotation = svd_solver.matrixV() * svd_solver.matrixU().transpose();

                  // Refer to: http://nghiaho.com/?page_id=671
                  mat_transf = Eigen::MatrixXf::Identity(4,4);
                  mat_transf.block(0,0,3,3) = mat_rotation;
                  mat_transf.block(0,3,3,1) = -mat_rotation * (*reference_corner_features_)[idx_refer].getCornerPosition ()
                          + (*scene_corner_features_)[idx_scene].getCornerPosition ();

                  if (!hasObtained (mat_transf))
                  {
                    transform_candidates_.push_back (mat_transf);
                  }
                }
              }
            }
            else
            {
              if ((*scene_circle_features_)[idx_scene].getNumPrincipalAxes () == 1)
              {
                // 只有一条主轴，以这条主轴为基准。
                for (std::size_t idx_axis = 0; idx_axis < (*reference_circle_features_)[idx_refer].getNumPrincipalAxes (); ++idx_axis)
                {
                  Eigen::Vector3f axis_scene = (*scene_circle_features_)[idx_scene].getPrincipalAxis (0);
                  Eigen::Vector3f axis_refer = (*reference_circle_features_)[idx_refer].getPrincipalAxis (idx_axis);
                  Eigen::Vector3f normal_scene = (*scene_circle_features_)[idx_scene].getNormal ();
                  Eigen::Vector3f normal_refer = (*reference_circle_features_)[idx_refer].getNormal ();

                  Eigen::Vector3f y_axis_scene = normal_scene.cross (axis_scene);
                  Eigen::Vector3f y_axis_refer = normal_refer.cross (axis_refer);

                  Eigen::Matrix3f mat_covariance = Eigen::MatrixXf::Zero(3,3);
                  mat_covariance += axis_refer * axis_scene.transpose ();
                  mat_covariance += normal_refer * normal_scene.transpose ();
                  mat_covariance += y_axis_refer * y_axis_scene.transpose ();

                  Eigen::JacobiSVD<Eigen::Matrix3f> svd_solver;
                  svd_solver.compute(mat_covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

                  Eigen::Matrix3f mat_rotation = svd_solver.matrixV() * svd_solver.matrixU().transpose();

                  // Refer to: http://nghiaho.com/?page_id=671
                  Eigen::Matrix4f mat_transf = Eigen::MatrixXf::Identity(4,4);
                  mat_transf.block(0,0,3,3) = mat_rotation;
                  mat_transf.block(0,3,3,1) = -mat_rotation * (*reference_corner_features_)[idx_refer].getCornerPosition ()
                          + (*scene_corner_features_)[idx_scene].getCornerPosition ();

                  // Detect if the transformation is valid.
                  if (!hasObtained (mat_transf))
                  {
                    transf_list.push_back (mat_transf);
                  }

                  y_axis_scene = (-normal_scene).cross (axis_scene);
                  y_axis_refer = normal_refer.cross (axis_refer);
                  mat_covariance = Eigen::MatrixXf::Zero(3,3);
                  mat_covariance += axis_refer * axis_scene.transpose ();
                  mat_covariance += normal_refer * (-normal_scene).transpose ();
                  mat_covariance += y_axis_refer * y_axis_scene.transpose ();

                  svd_solver.compute(mat_covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

                  mat_rotation = svd_solver.matrixV() * svd_solver.matrixU().transpose();

                  // Refer to: http://nghiaho.com/?page_id=671
                  mat_transf = Eigen::MatrixXf::Identity(4,4);
                  mat_transf.block(0,0,3,3) = mat_rotation;
                  mat_transf.block(0,3,3,1) = -mat_rotation * (*reference_corner_features_)[idx_refer].getCornerPosition ()
                          + (*scene_corner_features_)[idx_scene].getCornerPosition ();

                  // Detect if the transformation is valid.
                  if (!hasObtained (mat_transf))
                  {
                    transform_candidates_.push_back (mat_transf);
                  }
                }
              }
              else
              {
                // 没有主轴，绕发现方向转动。
                Eigen::Vector3f normal_scene = (*scene_circle_features_)[idx_scene].getNormal ();
                Eigen::Vector3f normal_refer = (*reference_circle_features_)[idx_refer].getNormal ();
                Eigen::Vector3f center_scene = (*scene_circle_features_)[idx_scene].getCenter ();
                Eigen::Vector3f center_refer = (*reference_circle_features_)[idx_refer].getCenter ();

                // Reverse the normal or not.
                // Eigen::Vector3f scene_normal;
                // if (reversed)
                //   scene_normal = -scene_feature.getNormal ();
                // else
                //   scene_normal = scene_feature.getNormal ();

                // Construct the base transformation.
                float theta_scene = std::acos (normal_scene.dot (Eigen::Vector3f (0.0, 0.0, 1.0)));
                Eigen::Vector3f axis_scene;
                if (theta_scene < 1.0E-6)
                {
                  theta_scene = 0.0;
                  axis_scene = Eigen::Vector3f (1.0, 0.0, 0.0);
                }
                else
                {
                  axis_scene = normal_scene.cross (Eigen::Vector3f (0.0, 0.0, 1.0));
                }
                Eigen::Affine3f affine_transf_scene = Eigen::Translation3f (center_scene)
                    * Eigen::AngleAxisf (theta_scene, axis_scene.normalized ());

                float theta_refer = std::acos (normal_refer.dot (Eigen::Vector3f (0.0, 0.0, 1.0)));
                Eigen::Vector3f axis_refer;
                if (theta_refer < 1.0E-6)
                {
                  theta_refer = 0.0;
                  axis_refer = Eigen::Vector3f (1.0, 0.0, 0.0);
                }
                else
                {
                  axis_refer = normal_refer.cross (Eigen::Vector3f (0.0, 0.0, 1.0));
                }
                Eigen::Affine3f affine_transf_refer = Eigen::Translation3f (center_refer)
                    * Eigen::AngleAxisf (theta_refer, axis_refer.normalized ());

                int num_rotation = std::floor (2*PI / resolution_);
                std::vector<Eigen::Matrix4Xf> inner_transf_list;
                std::vector<float> inner_objective_list;
                for (int idx_roation = 0; idx_roation <= num_rotation; ++idx_roation)
                {
                  Eigen::Affine3f affine_transf_scene_new = affine_transf_scene
                      * Eigen::AngleAxisf (float (idx_roation)*resolution_, normal_scene);
                  Eigen::Affine3f transformation = affine_transf_refer * affine_transf_scene_new.inverse ();

                  Eigen::Matrix4f mat_transf = transformation.matrix ();
                  if (!hasObtained (mat_transf))
                    transform_candidates_.push_back(mat_transf);
                }

                // Reverse the direction of the normal.
                normal_scene = -normal_scene;
                // Construct the base transformation.
                theta_scene = std::acos (normal_scene.dot (Eigen::Vector3f (0.0, 0.0, 1.0)));
                if (theta_scene < 1.0E-6)
                {
                  theta_scene = 0.0;
                  axis_scene = Eigen::Vector3f (1.0, 0.0, 0.0);
                }
                else
                {
                  axis_scene = normal_scene.cross (Eigen::Vector3f (0.0, 0.0, 1.0));
                }
                affine_transf_scene = Eigen::Translation3f (center_scene)
                    * Eigen::AngleAxisf (theta_scene, axis_scene.normalized ());

                for (int idx_roation = 0; idx_roation <= num_rotation; ++idx_roation)
                {
                  Eigen::Affine3f affine_transf_scene_new = affine_transf_scene
                      * Eigen::AngleAxisf (float (idx_roation)*resolution_, normal_scene);
                  Eigen::Affine3f transformation = affine_transf_refer * affine_transf_scene_new.inverse ();

                  Eigen::Matrix4f mat_transf = transformation.matrix ();
                  if (!hasObtained (mat_transf))
                    transform_candidates_.push_back(mat_transf);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      std::cout << "Error, no LGFeatures." << std::endl;
      return (false);
    }

    // Rectify candidates.
    rectifyCandidates ();
    float rate = objective_value_ / (static_cast<float>(filtered_point_cloud_->size ()) * 0.01);
    if (rate < rate_threshold_)
    {
      transf_matrix = rough_transform_;
      return (true);
    }
    else
    {
      return (false);
    }
  }

  bool
  LGFCorrespGroup::pairFeatures (const std::vector<float> & angle_seq_scene, const std::vector<float> & angle_seq_refer,
      std::vector<std::vector<pcl::Correspondence> > & angle_corresps_list)
  {
    angle_corresps_list = std::vector<std::vector<pcl::Correspondence> > ();

    // Combine and find the pair.
    // The number of angles in the scene_feature MUST be less than or equal to that in the refer_feature.
    if (angle_seq_scene.size() <= angle_seq_refer.size())
    {
      // Fully pair.
      int idx_start;
      for (std::size_t idx_refer = 0; idx_refer < angle_seq_refer.size(); ++idx_refer)
      {
        // Find the start index.
        if (std::abs(angle_seq_scene[0]-angle_seq_refer[idx_refer]) < angle_threshold_)
        {
          idx_start = idx_refer;
          // Start to match the angles.
          bool flag_matched = true;
          for (std::size_t k = 0; k < angle_seq_scene.size(); ++k)
          {
            float angle_scene = angle_seq_scene[k];
            float angle_refer = angle_seq_refer[(idx_start+k)%angle_seq_refer.size()];
            if (std::abs(angle_scene-angle_refer) > angle_threshold_)
            {
              flag_matched = false;
              break;
            }
          }

          if (flag_matched)
          {
            std::vector<pcl::Correspondence> corresp_list;
            for (std::size_t k = 0; k < angle_seq_scene.size(); ++k)
            {
              pcl::Correspondence corresp;
              corresp.index_query = k;
              corresp.index_match = (idx_start+k)%angle_seq_refer.size();
              corresp_list.push_back(corresp);
            }

            angle_corresps_list.push_back(corresp_list);
          }
        }
      }
    }

    if (angle_corresps_list.empty ())
      return (false);
    else
      return (true);
  }

  bool
  LGFCorrespGroup::hasObtained (const Eigen::Matrix4f & mat_transform)
  {
    bool has_obtained = false;
    for (std::size_t i = 0; i < transform_candidates_.size (); ++i)
    {
      Eigen::Matrix4f mat_error = mat_transform.transpose () * transform_candidates_[i];
      float error_angle = std::abs (std::acos (((mat_error(0,0)+mat_error(1,1)+mat_error(2,2))-1)/2.0));
      Eigen::Vector3f error_vector = mat_error.block(0,3,3,1);
      float error_position = std::sqrt (error_vector.dot (error_vector));
      if ((error_angle < angle_threshold_) && (error_position < distance_threshold_))
      {
        has_obtained = true;
      }
    }

    return (has_obtained);
  }

  bool
  LGFCorrespGroup::rectifyCandidates ()
  {
    IterativeClosestPoint icp;
    icp.setReferCloud (reference_point_cloud_);
    icp.setSceneCloud (filtered_point_cloud_);

    std::vector<float> objectives (transform_candidates_.size ());
    for (std::size_t i = 0; i < transform_candidates_.size (); ++i)
    {
      try
      {
        float objective_value = icp.calObjectiveValue (transform_candidates_[i]);
        objectives[i] = objective_value;
        // std::cout << "Objective value: " << objective_value << std::endl;
      }
      catch (char const * msg)
      {
        // std::cout << "Bad initial transformation. " << msg << std::endl;
        objectives[i] = 1E10;
      }
    }

    if (!objectives.empty ())
    {
      std::vector<std::size_t> order_indices (objectives.size ());
      std::iota(order_indices.begin (), order_indices.end (), 0);
      std::sort(order_indices.begin (), order_indices.end (), [&objectives](int idx_1, int idx_2)
              { return objectives[idx_1] < objectives[idx_2]; });

      objective_value_ = objectives[order_indices[0]];
      rough_transform_ = transform_candidates_[order_indices[0]];
    }
  }

} // namespace radi
