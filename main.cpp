#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>

#include "distance_measurer.h"
#include "kinect2_grabber.h"
#include "gui/main_window.h"


#include <QApplication>
// #include "gui/main_window.h"

// #include "vtkAutoInit.h"
// VTK_MODULE_INIT(vtkRenderingFreeType);

Q_DECLARE_METATYPE(pcl::PointCloud<pcl::PointXYZRGB>::Ptr)
Q_DECLARE_METATYPE(std::vector<int>)

int main (int argc, char * argv[])
{
  QApplication app (argc, argv);

  qRegisterMetaType<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>();
  qRegisterMetaType<std::vector<int> >();

  radi::MainWindow main_window;
  main_window.show ();
  return app.exec ();


  /*
  // boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud;
  // boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
  // K2G k2g(CPU);
  radi::Kinect2Grabber kinect2_grabber(radi::ProcessorType::OPENGL);
  std::cout << "getting cloud" << std::endl;
  kinect2_grabber.start();
  // cloud = k2g.getCloud();
  // kinect2_grabber.getPointCloud(cloud);
  kinect2_grabber.getPointCloud(cloud);
  // cloud = kinect2_grabber.getPointCloud();

  // k2g.printParameters();

  cloud->sensor_orientation_.w() = 0.0;
  cloud->sensor_orientation_.x() = 1.0;
  cloud->sensor_orientation_.y() = 0.0point_;
  cloud->sensor_orientation_.z() = 0.0;

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

  // PlySaver ps(cloud, false, false, k2g);
  // viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&ps);

  cv::Mat color, depth;

  while(!viewer->wasStopped()){

    viewer->spinOnce ();
    std::chrono::high_resolution_clock::time_point tnow = std::chrono::high_resolution_clock::now();

    // k2g.get(color, depth, cloud);
    // kinect2_grabber.get(color, depth, cloud);
    kinect2_grabber.getPointCloud(cloud);
    // Showing only color since depth is float and needs conversion
    // cv::imshow("color", color);
    // int c = cv::waitKey(1);

    std::chrono::high_resolution_clock::time_point tpost = std::chrono::high_resolution_clock::now();
    std::cout << "delta " << std::chrono::duration_cast<std::chrono::duration<double>>(tpost-tnow).count() * 1000 << std::endl;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->updatePointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");

  }

  // k2g.shutDown();
  kinect2_grabber.shutDown();


  radi::DistanceMeasurer dist;
  //
  // Specify CVSFeature(s) in the reference model.
  Eigen::Matrix4f mat_camera = Eigen::Matrix4Xf::Identity(4,4);
  // mat_camera(0,0) = -0.7071;
  // mat_camera(0,1) = -0.7071;
  // mat_camera(0,2) = 0.0;
  // mat_camera(1,0) = -0.5;
  // mat_camera(1,1) = 0.5;
  // mat_camera(1,2) = 0.7071;
  // mat_camera(2,0) = -0.5;
  // mat_camera(2,1) = 0.5;
  // mat_camera(2,2) = -0.7071;
  mat_camera(0,0) = 0.7071;
  mat_camera(0,1) = -0.5;
  mat_camera(0,2) = 0.5;
  mat_camera(1,0) = 0.7071;
  mat_camera(1,1) = 0.5;
  mat_camera(1,2) = -0.5;
  mat_camera(2,0) = 0.0;
  mat_camera(2,1) = 0.7071;
  mat_camera(2,2) = 0.7071;

  mat_camera(0,3) = 2.0;
  mat_camera(1,3) = -2.0;
  mat_camera(2,3) = 2.0;

  Eigen::Matrix4f inv_mat_camera = mat_camera.inverse ();


  // Specify CCN Features in the model.
  Eigen::Matrix4f mat_camera_ccn = Eigen::Matrix4Xf::Identity(4,4);
  mat_camera_ccn(0,0) = 0.0;
  mat_camera_ccn(0,1) = 1.0;
  mat_camera_ccn(0,2) = 0.0;
  mat_camera_ccn(1,0) = -0.3752;
  mat_camera_ccn(1,1) = 0.0;
  mat_camera_ccn(1,2) = 0.927;
  mat_camera_ccn(2,0) = 0.927;
  mat_camera_ccn(2,1) = 0.0;
  mat_camera_ccn(2,2) = 0.3752;

  mat_camera_ccn(0,3) = 2.5;
  mat_camera_ccn(1,3) = 0.0;
  mat_camera_ccn(2,3) = 1.0;

  */
}

