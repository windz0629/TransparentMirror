#include <pcl/io/pcd_io.h>
#include "thread_capture.h"

namespace radi
{
  ThreadCapture::ThreadCapture (QMutex * mutex, Kinect2Grabber * kinect2_grabber)
      : mutex_(mutex), kinect2_grabber_(kinect2_grabber), is_debut_(true)
  { }

  ThreadCapture::~ThreadCapture ()
  {
    this->quit ();
    this->wait ();
  }

  void
  ThreadCapture::run ()
  {
    while (true)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());

      mutex_->lock ();
      kinect2_grabber_->getPointCloud (point_cloud);
      mutex_->unlock ();

      point_cloud->sensor_orientation_.w() = 0.0;
      point_cloud->sensor_orientation_.x() = 1.0;
      point_cloud->sensor_orientation_.y() = 0.0;
      point_cloud->sensor_orientation_.z() = 0.0;

      // if (is_debut_)
      // {
      //   is_debut_ = false;
      //   pcl::io::savePCDFileASCII<pcl::PointXYZRGB>("./Models/scene_kinect.pcd", *point_cloud);
      // }

      emit updateVTKKinect (point_cloud);
    }
  }

} // namespace radi
