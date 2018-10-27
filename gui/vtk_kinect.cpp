#include <QEvent>
#include <QKeyEvent>
#include <QApplication>
#include <pcl/io/pcd_io.h>
#include <vtkRenderWindow.h>
#include "vtk_kinect.h"

namespace radi
{
  VTKKinect::VTKKinect () : point_cloud_(new pcl::PointCloud<pcl::PointXYZRGB> ()),
      viewer_(new pcl::visualization::PCLVisualizer ("Kinect", false)), is_debut(true)
  {
    // viewer_->setBackgroundColor (0, 0, 0);

    this->SetRenderWindow (viewer_->getRenderWindow ());
    viewer_->setupInteractor (this->GetInteractor (), this->GetRenderWindow ());
    viewer_->addPointCloud (point_cloud_, "Kinect");
    this->update ();
  }

  VTKKinect::~VTKKinect ()
  { }

  void
  VTKKinect::updatePointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud)
  {
    point_cloud_ = point_cloud;
    viewer_->updatePointCloud (point_cloud_, "Kinect");
    if (is_debut)
    {
      is_debut = false;
      viewer_->resetCamera();
      // pcl::io::savePCDFile("./Models/scene_home_5.pcd", *point_cloud_);
    }

    this->update ();
  }

} // namespace radi
