/*
 * GUI -- Window used show the 3D model of the scene.
 */

#ifndef RADI_VTK_MODEL_H_
#define RADI_VTK_MODEL_H_

#include <Eigen/Core>
#include <QString>
#include <QMap>
#include <QColor>
#include <QVTKWidget.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace radi
{
  class VTKModel : public QVTKWidget
  {
      Q_OBJECT

    public:
      VTKModel ();
      ~VTKModel ();

      void
      setModelFile (const QString & file_model);

    public slots:
      void
      updateModel (QMap<QString, Eigen::Matrix4f> map_transform);

    signals:
      void
      updateLogWindow (QString message);

      void
      sendPointClouds (QMap<QString, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> map_point_cloud);

    private:
      QString file_model_;
      QList<QString> object_names_;
      QMap<QString, QString> map_files_;
      QMap<QString, QColor> map_color_;
      QMap<QString, Eigen::Matrix4f> map_frame_;
      QMap<QString, bool> map_visible_;
      QMap<QString, bool> map_in_viewer_;
      QMap<QString, pcl::PolygonMesh> map_polygon_mesh_;
      QMap<QString, pcl::PointCloud<pcl::PointXYZRGB>::Ptr > map_point_cloud_;
      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;

      // radi::ColladaReader collada_reader_;

  }; // class ThreadCapture

} // namespace radi

#endif
