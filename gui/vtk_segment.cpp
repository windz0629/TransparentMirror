#include <pcl/filters/extract_indices.h>
#include <vtkRenderWindow.h>
#include "vtk_segment.h"

namespace radi
{
  VTKSegment::VTKSegment () : point_cloud_(new pcl::PointCloud<pcl::PointXYZRGB> ()),
      viewer_(new pcl::visualization::PCLVisualizer ("Segmentation", false)), is_debut_(true)
  {
    viewer_->setBackgroundColor (1.0, 1.0, 1.0);

    this->SetRenderWindow (viewer_->getRenderWindow ());
    viewer_->setupInteractor (this->GetInteractor (), this->GetRenderWindow ());
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_);
    viewer_->addPointCloud (point_cloud_, rgb, "cluster");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_edge_points (new pcl::PointCloud<pcl::PointXYZRGB> ());
    viewer_->addPointCloud<pcl::PointXYZRGB>(cloud_edge_points, "edge points");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "edge points");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 0.0f, 0.0f, "edge points");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_board_points (new pcl::PointCloud<pcl::PointXYZRGB> ());
    viewer_->addPointCloud<pcl::PointXYZRGB>(cloud_board_points, "board points");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "board points");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 1.0f, 0.0f, "board points");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_feature_points (new pcl::PointCloud<pcl::PointXYZRGB> ());
    viewer_->addPointCloud<pcl::PointXYZRGB>(cloud_feature_points, "feature points");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "feature points");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0f, 0.0f, 1.0f, "feature points");

    this->update ();
  }

  VTKSegment::~VTKSegment ()
  { }

  void
  VTKSegment::updatePointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud)
  {
    point_cloud_ = point_cloud;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_);
    viewer_->updatePointCloud (point_cloud_, rgb, "cluster");
    viewer_->resetCamera();

    this->update ();
  }

  void
  VTKSegment::updatePointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud,
      const std::vector<int> & edge_point_indices, const std::vector<int> & board_point_indices)
  {
    point_cloud_ = point_cloud;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_);
    viewer_->updatePointCloud (point_cloud_, rgb, "cluster");
    viewer_->resetCamera();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_edge_points(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ExtractIndices<pcl::PointXYZRGB> extracter;
    extracter.setInputCloud(this->point_cloud_);
    extracter.setIndices(boost::make_shared<std::vector<int> > (edge_point_indices));
    extracter.setNegative(false);
    extracter.filter(*cloud_edge_points);
    viewer_->updatePointCloud<pcl::PointXYZRGB>(cloud_edge_points, "edge points");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_board_points(new pcl::PointCloud<pcl::PointXYZRGB>);
    extracter.setIndices(boost::make_shared<std::vector<int> > (board_point_indices));
    extracter.setNegative(false);
    extracter.filter(*cloud_board_points);
    viewer_->updatePointCloud<pcl::PointXYZRGB>(cloud_board_points, "board points");

    this->update ();
  }

  void
  VTKSegment::showFeaturePoints (const std::vector<int> & feature_point_indices)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_feature_points(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ExtractIndices<pcl::PointXYZRGB> extracter;
    extracter.setInputCloud(this->point_cloud_);
    extracter.setIndices(boost::make_shared<std::vector<int> > (feature_point_indices));
    extracter.setNegative(false);
    extracter.filter(*cloud_feature_points);
    viewer_->updatePointCloud<pcl::PointXYZRGB>(cloud_feature_points, "feature points");

    this->update ();
  }

} // namespace radi
