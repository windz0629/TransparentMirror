/*
 * GUI -- Window used to show the segmented object.
 */

#ifndef RADI_VTK_SEGMENT_H_
#define RADI_VTK_SEGMENT_H_

#include <QVTKWidget.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace radi
{
  class VTKSegment : public QVTKWidget
  {
      Q_OBJECT

    public:
      VTKSegment ();
      ~VTKSegment ();

    public slots:
      void
      updatePointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud);

      void
      updatePointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud,
          const std::vector<int> & edge_point_indices, const std::vector<int> & board_point_indices);

      void
      showFeaturePoints (const std::vector<int> & feature_point_indices);

    private:
      bool is_debut_;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_;
      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;

  }; // class ThreadCapture

} // namespace radi

#endif
