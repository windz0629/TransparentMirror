#include <boost/make_shared.hpp>
#include <Eigen/Eigenvalues>

#include <pcl/octree/octree_search.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "feature_point_extractor.h"
#include "lgf_estimator.h"

namespace radi
{
  void
  removeIndices (std::vector<int> & source_indices, const std::vector<int> & removed_indices);

  void
  distanceLine2Line (const pcl::ModelCoefficients & line_1, const pcl::ModelCoefficients & line_2,
      float & distance, Eigen::Vector3f & point_1, Eigen::Vector3f & point_2);

  CEFEstimator::CEFEstimator() : radius_(0.05), distance_(0.2), min_num_points_(20), min_num_edges_(2)
  { }

  CEFEstimator::~CEFEstimator()
  { }

  void CEFEstimator::setInputCloud(const PointCloudConstPtr & point_cloud)
  {
    point_cloud_ = point_cloud;
  }

  void
  CEFEstimator::setCornerIndices (const std::vector<int> & corner_indices)
  {
    corner_indices_ = corner_indices;
  }

  void
  CEFEstimator::setFeaturePointIndices (const std::vector<int> & feature_point_indices)
  {
    feature_point_indices_ = feature_point_indices;
  }

  void CEFEstimator::setRadius(float radius)
  {
    radius_ = radius;
  }

  void CEFEstimator::setMinNumEdges(std::size_t min_num_edges)
  {
    min_num_edges_ = min_num_edges;
  }

  void CEFEstimator::esimate(std::vector<CornerEdgesFeature> & corner_edges_feature_list)
  {
    pcl::SACSegmentation<pcl::PointXYZRGB> sac_segment;
    sac_segment.setInputCloud (this->point_cloud_);
    sac_segment.setModelType (pcl::SACMODEL_LINE);
    sac_segment.setMethodType (pcl::SAC_RANSAC);
    sac_segment.setMaxIterations (100);
    sac_segment.setOptimizeCoefficients(true);
    sac_segment.setDistanceThreshold (0.01);

    std::vector<pcl::ModelCoefficients> line_list;
    while (feature_point_indices_.size () > min_num_points_)
    {
      sac_segment.setIndices (boost::make_shared<std::vector<int> >(feature_point_indices_));
      pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
      sac_segment.segment (*inliers, *coefficients);

      if (inliers->indices.size() >= min_num_points_)
      {
        std::cout << "Number of inliers: " << inliers->indices.size() << std::endl;
        std::cout << "Model coefficients: " << *coefficients << std::endl;
        line_list.push_back(*coefficients);

        // // Visualize points on a circle.
        // pcl::PointCloud<pcl::PointXYZ>::Ptr circle(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::ExtractIndices<pcl::PointXYZ> extracter;
        // extracter.setInputCloud(this->point_cloud_);
        // extracter.setIndices(inliers);
        // extracter.setNegative(false);
        // extracter.filter(*circle);
        // pcl::visualization::PCLVisualizer viewer("Board Points");
        // viewer.addPointCloud(this->point_cloud_);
        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(circle, 0, 255, 0);
        // viewer.addPointCloud(circle, single_color, "Board points");
        // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Board points");
        // while (!viewer.wasStopped()) {
        //     viewer.spinOnce();
        // }
      }

      removeIndices(feature_point_indices_, inliers->indices);
    }

    // Detect corner points.
    std::vector<std::vector<ClosestPoints> > corner_candidate_list;
    for (int i = 0; i < line_list.size(); ++i)
    {
      for (int j = i+1; j < line_list.size(); ++j)
      {
        float distance;
        Eigen::Vector3f point_1;
        Eigen::Vector3f point_2;
        distanceLine2Line(line_list[i], line_list[j], distance, point_1, point_2);
        if (distance < 0.03)
        {
          ClosestPoints closest_points;
          closest_points.idx_line_1 = i;
          closest_points.idx_line_2 = j;
          closest_points.distance = distance;
          closest_points.point_1 = point_1;
          closest_points.point_2 = point_2;

          if (!corner_candidate_list.empty())
          {
            bool already_in = false;
            for (int idx_corner = 0; idx_corner < corner_candidate_list.size(); ++idx_corner)
            {
              if ((closest_points.point_1-corner_candidate_list[idx_corner][0].point_1).norm() < distance)
              {
                corner_candidate_list[idx_corner].push_back(closest_points);
                already_in = true;
                break;
              }

              if (!already_in)
              {
                std::vector<ClosestPoints> corner_candidate;
                corner_candidate.push_back(closest_points);
                corner_candidate_list.push_back(corner_candidate);
              }
            }

          }
          else
          {
            std::vector<ClosestPoints> corner_candidate;
            corner_candidate.push_back(closest_points);
            corner_candidate_list.push_back(corner_candidate);
          }
        }
        std::cout << "Distance: " << distance << std::endl;
        std::cout << "Point-1: \n" << point_1 << std::endl;
        std::cout << "Point-2: \n" << point_2 << std::endl;
      }
    }

    std::cout << "Number of corner candidats: " << corner_candidate_list.size() << std::endl;

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_point_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
    // pcl::ExtractIndices<pcl::PointXYZRGB> extractor;
    // extractor.setInputCloud (point_cloud_);
    // extractor.setIndices (boost::make_shared<std::vector<int> > (feature_point_indices_));
    // extractor.filter (*filtered_point_cloud);

    // pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree (0.01);
    // octree.setInputCloud (filtered_point_cloud);
    // octree.addPointsFromInputCloud ();
    // for (std::size_t idx_corner = 0; idx_corner < corner_indices_.size(); ++idx_corner)
    // {
    //   // Extract the neighborhood of the corner point, in which edge detection will be performed.
    //   pcl::PointXYZRGB corner ((*point_cloud_).points[corner_indices_[idx_corner]].x,
    //       (*point_cloud_).points[corner_indices_[idx_corner]].y,
    //       (*point_cloud_).points[corner_indices_[idx_corner]].z);
    //   std::vector<int> neighbor_indices;
    //   std::vector<float> neighbor_distances;
    //   octree.radiusSearch (corner, radius_, neighbor_indices, neighbor_distances);

    //   pcl::ExtractIndices<pcl::PointXYZRGB> extractor;
    //   extractor.setInputCloud (point_cloud_);
    //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr neighborhood (new pcl::PointCloud<pcl::PointXYZRGB> ());
    //   // Need to convert the indices from std::vector<int> to boost::shared_ptr<std::vector<int> >.
    //   extractor.setIndices (boost::make_shared<std::vector<int> > (neighbor_indices));
    //   extractor.filter (*neighborhood);

    //   std::cout << "Number of points in the neighborhood of corner point: " << neighborhood->size() << std::endl;

    //   // ToDo: Need to rectify the following algorithm to be more robust.
    //   // Categorize the edge points into different clusters.
    //   Eigen::Vector3f pos_corner (corner.x, corner.y, corner.z);
    //   std::vector<std::vector<int> > edge_candidates;
    //   // When encounter a new direction, add it into 'direction_references'.
    //   std::vector<Eigen::Vector3f> direction_references;
    //   for (std::size_t idx_edge = 0; idx_edge < neighborhood->size (); ++idx_edge)
    //   {
    //     const pcl::PointXYZRGB & point_edge = (*neighborhood).points[idx_edge];
    //     Eigen::Vector3f pos_edge_point (point_edge.x, point_edge.y, point_edge.z);
    //     Eigen::Vector3f vect_direction = pos_edge_point - pos_corner;
    //     vect_direction /= std::sqrt(vect_direction.dot(vect_direction));

    //     if (edge_candidates.empty())
    //     {
    //       edge_candidates.push_back (std::vector<int> ());
    //       edge_candidates[0].push_back (idx_edge);
    //       direction_references.push_back (vect_direction);
    //     }
    //     else
    //     {
    //       bool has_found = false;
    //       for (std::size_t idx_direction = 0; idx_direction < direction_references.size(); ++idx_direction)
    //       {
    //         if (1.0-vect_direction.dot(direction_references[idx_direction]) < 0.1)
    //         {
    //           has_found = true;
    //           edge_candidates[idx_direction].push_back (idx_edge);
    //           break;
    //         }
    //       }

    //       if (!has_found)
    //       {
    //         edge_candidates.push_back (std::vector<int> ());
    //         edge_candidates[edge_candidates.size()-1].push_back (idx_edge);
    //         direction_references.push_back (vect_direction);
    //       }
    //     }
    //   }

    //   // Construct cvs feature.
    //   CornerEdgesFeature corner_edges_feature;
    //   corner_edges_feature.setCorner (corner);
    //   // Number of the points in one edge should be larger than 5.
    //   for (std::size_t idx_feature = 0; idx_feature < edge_candidates.size(); ++idx_feature)
    //   {
    //     if (edge_candidates[idx_feature].size() >= 5)
    //     {
    //       corner_edges_feature.appendEdge (direction_references[idx_feature]);
    //     }
    //   }

    //   std::cout << "Number of edges: " << corner_edges_feature.getNumEdges () << std::endl;
    //   if (corner_edges_feature.getNumEdges() >= min_num_edges_) {
    //     corner_edges_feature.compute();

    //     // const std::vector<float> & angles = cvs_feature.getIncludedAngles();
    //     // std::cout << angles[0] << "  " << angles[1] << "  " << angles[2] << std::endl;

    //     corner_edges_feature_list.push_back (corner_edges_feature);
    //   }
    // }
  }

  float CEFEstimator::getRadius()
  {
    return (radius_);
  }

  std::size_t CEFEstimator::getMinNumEdges()
  {
    return (min_num_edges_);
  }


  CNFEstimator::CNFEstimator() : min_radius_(0.05), max_radius_(0.3), threshold_distance_(0.02), search_radius_(0.2),
      min_num_points_(10), threshold_position_(0.05), threshold_radius_(0.03), threshold_angle_(0.5)
  { }

  CNFEstimator::~CNFEstimator()
  { }

  void
  CNFEstimator::setInputCloud (const PointCloudConstPtr & point_cloud)
  {
    point_cloud_ = point_cloud;
  }

  void
  CNFEstimator::setFeaturePointIndices (const std::vector<int> & feature_point_indices)
  {
    feature_point_indices_ = feature_point_indices;
  }

  void
  CNFEstimator::setRadiusLimits (float min_radius, float max_radius)
  {
    min_radius_ = min_radius;
    max_radius_ = max_radius;
  }

  void
  CNFEstimator::setSearchRadius (float search_radius)
  {
    search_radius_ = search_radius;
  }

  void
  CNFEstimator::setMinNumPoints (int min_num_points)
  {
    min_num_points_ = min_num_points;
  }

  void
  CNFEstimator::setIdentityThresholds(float position, float radius, float angle)
  {
    threshold_position_ = position;
    threshold_radius_ = radius;
    threshold_angle_ = angle;
  }

  void CNFEstimator::esimate(std::vector<CircleNormalFeature> & circle_normal_feature_list)
  {
    // Classify the board points.
    pcl::SACSegmentation<pcl::PointXYZRGB> sac_segment;
    sac_segment.setInputCloud(this->point_cloud_);
    sac_segment.setModelType(pcl::SACMODEL_CIRCLE3D);
    sac_segment.setMethodType(pcl::SAC_RANSAC);
    sac_segment.setMaxIterations(100);
    sac_segment.setRadiusLimits(min_radius_, max_radius_);
    sac_segment.setDistanceThreshold(threshold_distance_);

    // ToDo: SACSegmentation是否会最后返回一个圆？

    std::vector<pcl::ModelCoefficients> circle_list;
    while (feature_point_indices_.size () > min_num_points_)
    {
      pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
      sac_segment.setIndices (boost::make_shared<std::vector<int> >(feature_point_indices_));

      // pcl::search::KdTree<pcl::PointXYZ>::Ptr search (new pcl::search::KdTree<pcl::PointXYZ>);
      // search->setInputCloud(this->point_cloud_);
      // sac_segment.setSamplesMaxDist(search_radius_, search);

      sac_segment.segment (*inliers, *coefficients);

      if (inliers->indices.size() >= min_num_points_)
      {
        // std::cout << "Number of inliers: " << inliers->indices.size() << std::endl;
        // std::cout << "Model coefficients: " << *coefficients << std::endl;
        if (~isInCircleList(*coefficients, circle_list))
        {
          circle_list.push_back(*coefficients);

          CircleNormalFeature circle_normal_feature;
          Eigen::Vector3f center ((*coefficients).values[0], (*coefficients).values[1], (*coefficients).values[2]);
          float radius = (*coefficients).values[3];
          Eigen::Vector3f normal ((*coefficients).values[4], (*coefficients).values[5], (*coefficients).values[6]);
          circle_normal_feature.setCenter (center);
          circle_normal_feature.setRadius (radius);
          circle_normal_feature.setNormal (normal);
          circle_normal_feature_list.push_back (circle_normal_feature);
        }

        // // Visualize points on a circle.
        // pcl::PointCloud<pcl::PointXYZ>::Ptr circle(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::ExtractIndices<pcl::PointXYZ> extracter;
        // extracter.setInputCloud(this->point_cloud_);
        // extracter.setIndices(inliers);
        // extracter.setNegative(false);
        // extracter.filter(*circle);
        // pcl::visualization::PCLVisualizer viewer("Board Points");
        // viewer.addPointCloud(this->point_cloud_);
        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(circle, 0, 255, 0);
        // viewer.addPointCloud(circle, single_color, "Board points");
        // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Board points");
        // while (!viewer.wasStopped()) {
        //     viewer.spinOnce();
        // }
      }

      removeIndices(feature_point_indices_, inliers->indices);
    }
  }

  float
  CNFEstimator::getMinRadius ()
  {
    return (min_radius_);
  }

  std::size_t
  CNFEstimator::getMinNumPoints ()
  {
    return (min_num_points_);
  }

  const pcl::IndicesConstPtr
  CNFEstimator::getIndices ()
  {
    return (indices_);
  }

  bool
  CNFEstimator::isInCircleList(const pcl::ModelCoefficients & coefficients, const std::vector<pcl::ModelCoefficients> & circle_list)
  {
    Eigen::Vector3f pos_center (coefficients.values[0], coefficients.values[1], coefficients.values[2]);
    float radius = coefficients.values[3];
    Eigen::Vector3f normal (coefficients.values[4], coefficients.values[5], coefficients.values[6]);

    for (int idx_circle = 0; idx_circle < circle_list.size(); ++idx_circle)
    {
      const pcl::ModelCoefficients & coeff_exist = circle_list[idx_circle];
      Eigen::Vector3f pos_center_exist (coeff_exist.values[0], coeff_exist.values[1], coeff_exist.values[2]);
      float radius_exist = coeff_exist.values[3];
      Eigen::Vector3f normal_exist (coeff_exist.values[4], coeff_exist.values[5], coeff_exist.values[6]);

      float err_position = std::abs(pos_center.norm() - pos_center_exist.norm());
      float err_radius = std::abs(radius - radius_exist);
      float err_angle = std::acos(std::abs(normal.dot(normal_exist)));

      if ((err_position < threshold_position_) && (err_radius < threshold_radius_) && (err_angle < threshold_angle_))
      {
        return true;
      }
    }

    return false;
  }

  void
  removeIndices (std::vector<int> & source_indices, const std::vector<int> & removed_indices)
  {
    std::vector<int> full_indices = source_indices;
    source_indices = std::vector<int> (full_indices.size() - removed_indices.size());
    int iCount = 0;
    for (int idx_full = 0; idx_full < full_indices.size(); ++idx_full)
    {
      bool flag_in = false;
      for (int idx_removed = 0; idx_removed < removed_indices.size(); ++idx_removed)
      {
        if (full_indices[idx_full] == removed_indices[idx_removed])
        {
          flag_in = true;
          break;
        }
      }

      if (!flag_in)
      {
        source_indices[iCount] = full_indices[idx_full];
        iCount++;
      }
    }
  }

  void
  distanceLine2Line (const pcl::ModelCoefficients & line_1, const pcl::ModelCoefficients & line_2,
      float & distance, Eigen::Vector3f & point_1, Eigen::Vector3f & point_2)
  {
    Eigen::Vector3f u (line_1.values[3], line_1.values[4], line_1.values[5]);
    Eigen::Vector3f v (line_2.values[3], line_2.values[4], line_2.values[5]);
    Eigen::Vector3f w_0 (line_1.values[0]-line_2.values[0], line_1.values[1]-line_2.values[1], line_1.values[2]-line_2.values[2]);

    float a = u.dot (u);
    float b = u.dot (v);
    float c = v.dot (v);
    float d = u.dot (w_0);
    float e = v.dot (w_0);

    float s_c = (b*e-c*d) / (a*c-b*b);
    float t_c = (a*e-b*d) / (a*c-b*b);

    distance = (w_0+s_c*u-t_c*v).norm();
    point_1 = Eigen::Vector3f (line_1.values[0], line_1.values[1], line_1.values[2]) + s_c*u;
    point_2 = Eigen::Vector3f (line_2.values[0], line_2.values[1], line_2.values[2]) + t_c*v;
  }

} // namespace radi
