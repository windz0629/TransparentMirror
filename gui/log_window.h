/*
 * GUI -- Log window for displaying messages.
 */

#ifndef RADI_LOG_WINDOW_H_
#define RADI_LOG_WINDOW_H_

#include <QTextEdit>
#include <QMutex>
#include "../kinect2_grabber.h"

namespace radi
{
  class LogWindow : public QTextEdit
  {
    public:
      LogWindow ();
      ~LogWindow ();

  }; // class ThreadCapture

} // namespace radi

#endif
