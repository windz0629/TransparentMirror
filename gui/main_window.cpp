#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL);

#include <QEvent>
#include <QKeyEvent>
#include <QWidget>
#include <QGridLayout>
#include <QApplication>
#include <QDesktopWidget>

#include <vtkSmartPointer.h>
#include <vtkPLYReader.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>

#include "main_window.h"

namespace radi
{
  MainWindow::MainWindow ()
  {
    file_model_ = "";
    mutex_ = new QMutex ();
    kinect2_grabber_ = new Kinect2Grabber (radi::ProcessorType::OPENGL);

    thread_capture_ = new ThreadCapture (mutex_, kinect2_grabber_);
    thread_recognition_ = new ThreadRecognition (mutex_, kinect2_grabber_);
    vtk_kinect_ = new VTKKinect ();
    vtk_segment_ = new VTKSegment ();
    vtk_model_ = new VTKModel ();
    log_window_ = new LogWindow ();


    setWindowParameters ();
    arrangeWidgets ();
    bindSignalSlots ();

    if (kinect2_grabber_->isOpen ())
    {
      log_window_->append ("Kinect is open.");
    }
    else
    {
      log_window_->append ("[Error] Fail to open Kinect.");
    }

    vtk_model_->setModelFile ("Models/scene.scn");
    thread_recognition_->setFeatureFile ("Models/model_features.lgf");

    kinect2_grabber_->start ();
    kinect2_grabber_->disableLog ();
    thread_capture_->start ();
    thread_recognition_->start ();
  }

  MainWindow::~MainWindow ()
  {
    kinect2_grabber_->shutDown ();
  }

  void
  MainWindow::setWindowParameters ()
  {
    this->setWindowTitle ("Transparent Mirror");

    int width = 800;
    int height = 600;

    QDesktopWidget * desktop = QApplication::desktop();
    int screen_width = desktop->width ();
    int screen_height = desktop->height ();

    int pos_x = (screen_width - width) / 2;
    int pos_y = (screen_height - height) / 2;

    this->resize(width, height);
    this->move(pos_x, pos_y);
  }

  void
  MainWindow::arrangeWidgets ()
  {
    QWidget * central_widget = new QWidget ();
    QGridLayout * layout_main = new QGridLayout ();
    layout_main->addWidget(vtk_kinect_, 0, 0);
    layout_main->addWidget (vtk_model_, 0, 1);
    layout_main->addWidget (vtk_segment_, 1, 0);
    layout_main->addWidget (log_window_, 1, 1);
    central_widget->setLayout(layout_main);
    this->setCentralWidget (central_widget);
  }

  void
  MainWindow::bindSignalSlots ()
  {
    connect (thread_capture_, SIGNAL(updateVTKKinect(pcl::PointCloud<pcl::PointXYZRGB>::Ptr)),
        vtk_kinect_, SLOT(updatePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr)));

    connect (thread_recognition_, SIGNAL(updateVTKSegment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr)),
        vtk_segment_, SLOT(updatePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr)));
    connect (thread_recognition_, SIGNAL(updateVTKSegment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, const std::vector<int> &, const std::vector<int> &)),
        vtk_segment_, SLOT(updatePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, const std::vector<int> &, const std::vector<int> &)));
    connect (thread_recognition_, SIGNAL(showFeaturePointsInVTKSegment(const std::vector<int> & )),
        vtk_segment_, SLOT(showFeaturePoints(const std::vector<int> &)));
    connect (thread_recognition_, SIGNAL(updateLogWindow(QString)), log_window_, SLOT(append(QString)));

    connect (vtk_model_, SIGNAL(updateLogWindow(QString)), log_window_, SLOT(append(QString)));
    connect (vtk_model_, SIGNAL(sendPointClouds(QMap<QString,pcl::PointCloud<pcl::PointXYZRGB>::Ptr>)),
        thread_recognition_, SLOT(receivePointClouds(QMap<QString,pcl::PointCloud<pcl::PointXYZRGB>::Ptr>)));
  }

} // namespace radi
