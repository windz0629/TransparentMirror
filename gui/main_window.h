/*
 * GUI -- Main Window.
 */

#ifndef MIRROR_MAIN_WINDOW_H_
#define MIRROR_MAIN_WINDOW_H_

#include <QMainWindow>
#include <QMutex>
#include <QTextEdit>
#include <QString>

#include "../kinect2_grabber.h"
#include "vtk_kinect.h"
#include "vtk_segment.h"
#include "vtk_model.h"
#include "log_window.h"
#include "../thread/thread_capture.h"
#include "../thread/thread_recognition.h"

namespace radi
{
  class MainWindow : public QMainWindow
  {
      Q_OBJECT

    public:
      MainWindow ();
      ~MainWindow ();

      void
      setModel (const QString & file_model);

    private:
      ThreadCapture * thread_capture_;
      ThreadRecognition * thread_recognition_;

      VTKKinect * vtk_kinect_;
      VTKSegment * vtk_segment_;
      VTKModel * vtk_model_;
      LogWindow * log_window_;

      QString file_model_;
      QMutex * mutex_;
      Kinect2Grabber * kinect2_grabber_;

      void
      setWindowParameters ();

      void
      arrangeWidgets ();

      void
      bindSignalSlots ();

  }; // class MainWindow

} // namespace radi

#endif
