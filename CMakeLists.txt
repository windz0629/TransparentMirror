cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(TransparentMirror)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Specify the CMAKE directories of the 3rd-party libraries with particular versions.
# set(AtHome ON)
set(AtHome OFF)
if(AtHome)
    set(VTK_DIR "/usr/local/vtk-7.1.1/lib/cmake/vtk-7.1/")
    set(PCL_DIR "/usr/local/pcl-1.8.0/share/pcl-1.8/")
    set(freenect2_DIR "/usr/local/freenect2/lib/cmake/freenect2/")
    set(Qt5Core_DIR "/usr/local/Qt5.7.1/5.7/gcc_64/lib/cmake/Qt5Core/")
    set(Qt5Widgets_DIR "/usr/local/Qt5.7.1/5.7/gcc_64/lib/cmake/Qt5Widgets/")
    set(Qt5_LIBRARY_DIRS "/usr/local/Qt5.7.1/5.7/gcc_64/lib/")
else()
    message(INFO "  Not at home.")
    set(VTK_DIR "/usr/local/vtk-7.1.1/lib/cmake/vtk-7.1/")
    set(PCL_DIR "/usr/local/pcl-1.8.0/share/pcl-1.8/")
    set(freenect2_DIR "/usr/local/freenect2/lib/cmake/freenect2/")
    set(Qt5Core_DIR "/usr/local/Qt5.9.6/5.9.6/gcc_64/lib/cmake/Qt5Core/")
    set(Qt5Widgets_DIR "/usr/local/Qt5.9.6/5.9.6/gcc_64/lib/cmake/Qt5Widgets/")
    set(Qt5_LIBRARY_DIRS "/usr/local/Qt5.9.6/5.9.6/gcc_64/lib/")
    set(Qt5Core_INCLUDE_DIRS "/usr/local/Qt5.9.6/5.9.6/gcc_64/include/QtCore/")
    set(Qt5Widgets_INCLUDE_DIRS "/usr/local/Qt5.9.6/5.9.6/gcc_64/include/QtWidgets/")
endif()

set(AtRADI OFF)
if (AtRADI)
  set(freenect2_DIR "/opt/freenect2/lib/cmake/freenect2/")
endif()

message(INFO ${Qt5_LIBRARY_DIRS})

# Set compile features
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC ${CMAKE_CXX_FLAGS}")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -std=c++11 -gencode arch=compute_60,code=sm_60;-G;-g)
if(AtHome)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_61,code=sm_61)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_35,code=sm_35)
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

# find_package(VTK 7.1 REQUIRED CONFIG)
find_package(VTK REQUIRED CONFIG)
include(${VTK_USE_FILE})
# Fix the bug in VTK which could cause compilation error.
get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
    list(APPEND vtk_flags ${it})
    endif()
endforeach()
foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

find_package(PCL 1.8 REQUIRED CONFIG)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)
find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5Core REQUIRED CONFIG)
find_package(Qt5Widgets REQUIRED CONFIG)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${freenect2_INCLUDE_DIRS})
include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
link_directories(${Qt5_LIBRARY_DIRS})

qt5_wrap_cpp(QT_MOC_HEADERS
    gui/main_window.h
    gui/vtk_kinect.h
    gui/vtk_segment.h
    gui/vtk_model.h
    gui/log_window.h
    thread/thread_capture.h
    thread/thread_recognition.h
)

set(SRC_LIST
    main.cpp
    icp.cpp
    segmentor.cpp
    # cuda/feature_point_extractor.cu
    # cuda/feature_point_extractor.h
    feature_point_extractor.cpp
    feature_point_extractor.h
    local_geometric_feature.cpp
    lgf_estimator.cpp
    lgf_corresp_group.cpp
    distance_measurer.cu
    distance_measurer.h
    kinect2_grabber.cpp
    gui/main_window.cpp
    gui/vtk_kinect.cpp
    gui/vtk_segment.cpp
    gui/vtk_model.cpp
    gui/log_window.cpp
    thread/thread_capture.cpp
    thread/thread_recognition.cpp
)

# cuda_add_executable(transparent_mirror ${SRC_LIST} ${QT_MOC_HEADERS})
cuda_add_executable(transparent_mirror ${SRC_LIST})
# add_executable(transparent_mirror ${SRC_LIST})
target_link_libraries(transparent_mirror ${PCL_LIBRARIES})
target_link_libraries(transparent_mirror ${BOOST_LIBRARIES})
target_link_libraries(transparent_mirror ${freenect2_LIBRARIES})
target_link_libraries(transparent_mirror ${OpenCV_LIBS})
target_link_libraries(transparent_mirror ${VTK_LIBRARIES})
# target_link_libraries(transparent_mirror Qt5Core)
target_link_libraries(transparent_mirror Qt5::Core)
# target_link_libraries(transparent_mirror Qt5Widgets)
target_link_libraries(transparent_mirror Qt5::Widgets)

message(STATUS "CMake Done!")
