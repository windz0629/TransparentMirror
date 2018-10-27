/*
 * GUI -- Thread for recognizing the pose of the objects.
 */

#ifndef MIRROR_THREAD_RECOGNITION_H_
#define MIRROR_THREAD_RECOGNITION_H_

#include <QThread>
#include <QMutex>
#include <QString>
#include <QMap>
#include <QColor>
#include <QTime>
#include <pcl/point_types.h>
#include "../kinect2_grabber.h"
#include "../local_geometric_feature.h"

namespace radi
{
  class ThreadRecognition : public QThread
  {
      Q_OBJECT

    public:
      ThreadRecognition (QMutex * mutex, Kinect2Grabber * kinect2_grabber);
      ~ThreadRecognition ();

      void
      setFeatureFile (const QString & file_feature);

      void
      run ();

    public slots:
      void
      receivePointClouds (QMap<QString, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> map_point_cloud);

    signals:
      void
      updateLogWindow (QString message);

      void
      updateVTKSegment (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud);

      void
      updateVTKSegment (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud,
          const std::vector<int> edge_point_indices, const std::vector<int> board_point_indices);

      void
      showFeaturePointsInVTKSegment (const std::vector<int> & feature_point_indices);

      void
      updateVTKModel (std::vector<QString> names, std::vector<Eigen::Matrix4f> mat_transforms);

    private:
      QMutex * mutex_;
      Kinect2Grabber * kinect2_grabber_;
      QTime * timer_;

      QString file_feature_;
      QMap<QString, QColor> map_color_;
      QMap<QString, std::vector<radi::CornerEdgesFeature> > map_corner_;
      QMap<QString, std::vector<radi::CircleNormalFeature> > map_circle_;
      QMap<QString, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> map_point_cloud_;

  }; // class Recognition

} // namespace radi

#endif
