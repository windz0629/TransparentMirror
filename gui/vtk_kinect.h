/*
 * GUI -- Window used to show the scene captured by Kinect.
 */

#ifndef RADI_VTK_KINECT_H_
#define RADI_VTK_KINECT_H_

#include <QVTKWidget.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace radi
{
  class VTKKinect : public QVTKWidget
  {
      Q_OBJECT

    public:
      VTKKinect ();
      ~VTKKinect ();

    public slots:
      void
      updatePointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud);

    private:
      bool is_debut;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_;
      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;

  }; // class ThreadCapture

} // namespace radi

#endif
