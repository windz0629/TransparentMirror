/*
 * GUI -- Thread for capturing the picture and point cloud which are only used to show the scene.
 */

#ifndef MIRROR_THREAD_CAPTURE_H_
#define MIRROR_THREAD_CAPTURE_H_

#include <QThread>
#include <QMutex>
#include <pcl/point_types.h>
#include "../kinect2_grabber.h"

namespace radi
{
  class ThreadCapture : public QThread
  {
      Q_OBJECT

    public:
      ThreadCapture (QMutex * mutex, Kinect2Grabber * kinect2_grabber);
      ~ThreadCapture ();

      void
      run ();

    private:
      bool is_debut_;
      QMutex * mutex_;
      Kinect2Grabber * kinect2_grabber_;

    signals:
      void updateVTKKinect (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud);
  }; // class ThreadCapture

} // namespace radi

#endif
