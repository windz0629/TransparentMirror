#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/octree.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include "segmentor.h"

namespace radi
{
  Segmentor::Segmentor () : file_raw_scene_(""), raw_point_cloud_(new pcl::PointCloud<pcl::PointXYZRGB>()),
      pass_through_minimum_x_(-0.4), pass_through_maximum_x_(0.4),
      pass_through_minimum_y_(-0.4), pass_through_maximum_y_(0.4),
      pass_through_minimum_z_(0.6), pass_through_maximum_z_(1.0),
      remove_outlier_mean_k_(50), remove_outlier_stddev_(1.0),
      downsample_status_(false), downsample_leaf_size_x_(0.001), downsample_leaf_size_y_(0.001), downsample_leaf_size_z_(0.001),
      cluster_minimum_(100), cluster_maximum_(20000), cluster_tolerance_(0.005)
  { }

  Segmentor::~Segmentor ()
  { }

  void
  Segmentor::setRawPointCloud (const QString & file_raw_scene)
  {
    file_raw_scene_ = file_raw_scene;
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_raw_scene_.toStdString(), *raw_point_cloud_) == -1)
    {
      QString msg_error = QString("[Error] Couldn't read file '") + file_raw_scene_ + QString("'.");
      std::cout << msg_error.toStdString() << std::endl;
    }
  }

  void
  Segmentor::setRawPointCloud (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr raw_point_cloud)
  {
    pcl::copyPointCloud(*raw_point_cloud, *raw_point_cloud_);
  }

  void
  Segmentor::setPassThroughLimitsX (float minimum, float maximum)
  {
    pass_through_minimum_x_ = minimum;
    pass_through_maximum_x_ = maximum;
  }

  void
  Segmentor::setPassThroughLimitsY (float minimum, float maximum)
  {
    pass_through_minimum_y_ = minimum;
    pass_through_maximum_y_ = maximum;
  }

  void
  Segmentor::setPassThroughLimitsZ (float minimum, float maximum)
  {
    pass_through_minimum_z_ = minimum;
    pass_through_maximum_z_ = maximum;
  }

  void
  Segmentor::setRemoveOutlierMeanK (std::size_t mean_k)
  {
    remove_outlier_mean_k_ = mean_k;
  }

  void
  Segmentor::setRemoveOutlierStddev (float stddev)
  {
    remove_outlier_stddev_ = stddev;
  }

  void
  Segmentor::setDownsampleStatus (bool status)
  {
    downsample_status_ = status;
  }

  void
  Segmentor::setDownsampleLeafSize (float leaf_x, float leaf_y, float leaf_z)
  {
    downsample_leaf_size_x_ = leaf_x;
    downsample_leaf_size_y_ = leaf_y;
    downsample_leaf_size_z_ = leaf_z;
  }

  void
  Segmentor::setClusterSizes (std::size_t minimum, std::size_t maximum)
  {
    cluster_minimum_ = minimum;
    cluster_maximum_ = maximum;
  }

  void
  Segmentor::setClusterTolerance (float cluster_tolerance)
  {
    cluster_tolerance_ = cluster_tolerance;
  }

  void
  Segmentor::segment (std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cluster_list)
  {
    // Remove NAN data.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_nan_clean (new pcl::PointCloud<pcl::PointXYZRGB>());
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*raw_point_cloud_, *scene_nan_clean, indices);

    // Pass through.
    pcl::PassThrough<pcl::PointXYZRGB> pass_through;
    std::vector<int> indices_x;
    pass_through.setInputCloud(scene_nan_clean);
    pass_through.setFilterFieldName("x");
    pass_through.setFilterLimits(pass_through_minimum_x_, pass_through_maximum_x_);
    pass_through.setNegative(false);
    pass_through.filter(indices_x);

    std::vector<int> indices_xy;
    pass_through.setIndices(boost::make_shared<std::vector<int> >(indices_x));
    pass_through.setFilterFieldName("y");
    pass_through.setFilterLimits(pass_through_minimum_y_, pass_through_maximum_y_);
    pass_through.setNegative(false);
    pass_through.filter(indices_xy);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_pass_through (new pcl::PointCloud<pcl::PointXYZRGB>());
    pass_through.setIndices(boost::make_shared<std::vector<int> >(indices_xy));
    pass_through.setFilterFieldName("z");
    pass_through.setFilterLimits(pass_through_minimum_z_, pass_through_maximum_z_);
    pass_through.setNegative(false);
    pass_through.filter(*scene_pass_through);

    pcl::io::savePCDFile("scene_home_3_pass_through.pcd", *scene_pass_through);

    // Remove outliers.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> outlier_filter;
    outlier_filter.setInputCloud(scene_pass_through);
    outlier_filter.setMeanK(50);
    outlier_filter.setStddevMulThresh(1.0);
    outlier_filter.filter(*scene_filtered);

    // Downsampling. Downsample the origin point cloud and reduce the number of points.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_downsampled (new pcl::PointCloud<pcl::PointXYZRGB>());
    if (downsample_status_)
    {
      pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid;
      vox_grid.setInputCloud(scene_filtered);
      vox_grid.setLeafSize(downsample_leaf_size_x_, downsample_leaf_size_y_, downsample_leaf_size_z_);
      vox_grid.filter(*scene_downsampled);
    }

    // Segmentation
    // Segmentation without color.
    //plane segment
    pcl::SACSegmentation<pcl::PointXYZRGB> plane_seg;
    if (downsample_status_)
    {
      plane_seg.setInputCloud(scene_downsampled);
    }
    else
    {
      plane_seg.setInputCloud(scene_filtered);
    }
    pcl::ModelCoefficients::Ptr coeffPtr(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    plane_seg.setOptimizeCoefficients(true);
    plane_seg.setModelType(pcl::SACMODEL_PLANE);
    plane_seg.setMethodType(pcl::SAC_RANSAC);
    plane_seg.setDistanceThreshold(0.005);
    plane_seg.segment(*inliers,*coeffPtr);
    while (inliers->indices.size() < 5000)
    {
      plane_seg.segment(*inliers,*coeffPtr);
    }
    // if(inliers->indices.size()==0)
    //     std::cerr<<"WARNING: no plane extracted"<<std::endl;
    // else
    //   std::cout<<"plane extracted, point size: "<<inliers->indices.size()<<std::endl;

    //extract plane and scene-without-plane
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_no_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ExtractIndices<pcl::PointXYZRGB> extractor;
    extractor.setInputCloud(scene_filtered);
    // extractor.setInputCloud(scene_downsampled);
    extractor.setIndices(inliers);
    extractor.setNegative(true);
    extractor.filter(*scene_no_plane);
    std::cout << "scene extracted, point size: " << scene_no_plane->points.size() << std::endl;

    //euclidean cluster
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(scene_no_plane);
    std::vector<pcl::PointIndices> cluster_indices_list;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> clusterExtrac;
    clusterExtrac.setInputCloud(scene_no_plane);
    clusterExtrac.setSearchMethod(tree);
    clusterExtrac.setClusterTolerance(cluster_tolerance_);
    clusterExtrac.setMinClusterSize(cluster_minimum_);
    clusterExtrac.setMaxClusterSize(cluster_maximum_);
    clusterExtrac.extract(cluster_indices_list);
    if(cluster_indices_list.size()==0)
        std::cerr << "ERROR: No cluster extracted" << std::endl;
    else
      std::cout << "Extracted " << cluster_indices_list.size() << " clusters." << std::endl;

    // Segmentation with color.
    // std::vector<pcl::PointIndices> cluster_indices_list;
    // pcl::search::Search <pcl::PointXYZRGB>::Ptr kdtree =
    //       boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);
    // pcl::RegionGrowingRGB<pcl::PointXYZRGB> region_rgb;
    // region_rgb.setInputCloud (scene_downsampled);
    // region_rgb.setSearchMethod (kdtree);
    // region_rgb.setPointColorThreshold (10);
    // region_rgb.setRegionColorThreshold (10);
    // region_rgb.setMinClusterSize (500);
    // region_rgb.extract (cluster_indices_list);

    pcl::ExtractIndices<pcl::PointXYZRGB> indices_extractor;
    indices_extractor.setInputCloud(scene_no_plane);
    for (int i = 0; i < cluster_indices_list.size(); ++i)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
      indices_extractor.setIndices(boost::make_shared<std::vector<int> >(cluster_indices_list[i].indices));
      indices_extractor.setNegative(false);
      indices_extractor.filter(*point_cluster);

      cluster_list.push_back(point_cluster);
    }
  }

} // namespace radi
