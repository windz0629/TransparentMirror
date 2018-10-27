#include <iostream>

#include <QFile>
#include <QXmlStreamReader>

#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/octree.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include "thread_recognition.h"

#include "../feature_point_extractor.h"
// #include "cuda/feature_point_extractor.h"   // Use CUDA

#include "../segmentor.h"
#include "../lgf_estimator.h"
#include "../lgf_corresp_group.h"

#include <pcl/io/pcd_io.h>


namespace radi
{
  ThreadRecognition::ThreadRecognition (QMutex * mutex, Kinect2Grabber * kinect2_grabber)
      : mutex_ (mutex), kinect2_grabber_ (kinect2_grabber), timer_(new QTime)
  { }

  ThreadRecognition::~ThreadRecognition ()
  {
    this->quit ();
    this->wait ();
  }

  void
  ThreadRecognition::setFeatureFile (const QString & file_feature)
  {
    file_feature_ = file_feature;
    QFile file(file_feature_);
    if (!file.open(QFile::ReadOnly|QFile::Text))
    {
      emit updateLogWindow ("[Error] " + QString("File '%1' cannot be found!").arg(file_feature_));
      return;
    }

    QXmlStreamReader xml_reader;
    xml_reader.setDevice(&file);
    while (xml_reader.readNextStartElement())
    {
      if(xml_reader.name() == "lgf")
      {
        while (xml_reader.readNextStartElement())
        {
          if (xml_reader.name() == "object")
          {
            QString object_name = xml_reader.attributes().value("name").toString();
            this->map_corner_[object_name] = std::vector<radi::CornerEdgesFeature> ();
            this->map_circle_[object_name] = std::vector<radi::CircleNormalFeature> ();

            emit updateLogWindow("[INFO] Find features of object '" + object_name + "'.");

            while (xml_reader.readNextStartElement())
            {
              if (xml_reader.name() == "color")
              {
                QString color = xml_reader.attributes().value("rgb").toString();
                QVector<QString> color_list = color.split(" ", QString::SkipEmptyParts).toVector();
                this->map_color_[object_name] = QColor(color_list[0].toInt(),
                    color_list[1].toInt(), color_list[2].toInt());
                xml_reader.readElementText();
              }
              else if (xml_reader.name() == "cef")
              {
                radi::CornerEdgesFeature corner_feature;
                while (xml_reader.readNextStartElement ())
                {
                  if (xml_reader.name() == "corner")
                  {
                    QString content = xml_reader.readElementText();
                    QVector<QString> pos_list = content.split(" ", QString::SkipEmptyParts).toVector();
                    Eigen::Vector3f position = Eigen::Vector3f(pos_list[0].toFloat(),
                        pos_list[1].toFloat(), pos_list[2].toFloat());
                    corner_feature.setCorner (position);
                  }
                  else if (xml_reader.name() == "edge")
                  {
                    QString content = xml_reader.readElementText();
                    QVector<QString> edge_list = content.split(" ", QString::SkipEmptyParts).toVector();
                    Eigen::Vector3f edge = Eigen::Vector3f(edge_list[0].toFloat(),
                        edge_list[1].toFloat(), edge_list[2].toFloat());
                    corner_feature.appendEdge (edge);
                  }
                  else
                  {
                    xml_reader.skipCurrentElement();
                  }
                }

                this->map_corner_[object_name].push_back (corner_feature);
              }
              else if (xml_reader.name() == "cnf")
              {
                radi::CircleNormalFeature circle_feature;
                while (xml_reader.readNextStartElement ())
                {
                  if (xml_reader.name() == "center")
                  {
                    QString content = xml_reader.readElementText();
                    QVector<QString> center_list = content.split(" ", QString::SkipEmptyParts).toVector();
                    Eigen::Vector3f center = Eigen::Vector3f(center_list[0].toFloat(),
                        center_list[1].toFloat(), center_list[2].toFloat());
                    circle_feature.setCenter (center);
                  }
                  else if (xml_reader.name() == "radius")
                  {
                    QString content = xml_reader.readElementText();
                    circle_feature.setRadius (content.toFloat ());
                  }
                  else if (xml_reader.name() == "normal")
                  {
                    QString content = xml_reader.readElementText();
                    QVector<QString> normal_list = content.split(" ", QString::SkipEmptyParts).toVector();
                    Eigen::Vector3f normal = Eigen::Vector3f(normal_list[0].toFloat(),
                        normal_list[1].toFloat(), normal_list[2].toFloat());
                    circle_feature.setNormal (normal);
                  }
                  else if (xml_reader.name() == "principal_axis")
                  {
                    QString content = xml_reader.readElementText();
                    QVector<QString> axis_list = content.split(" ", QString::SkipEmptyParts).toVector();
                    Eigen::Vector3f axis = Eigen::Vector3f(axis_list[0].toFloat(),
                        axis_list[1].toFloat(), axis_list[2].toFloat());
                    circle_feature.appendPrincipalAxis (axis);
                  }
                  else
                  {
                    xml_reader.skipCurrentElement();
                  }
                }

                this->map_circle_[object_name].push_back (circle_feature);
              }
              else
              {
                xml_reader.skipCurrentElement();
              }
            }
          }
        }
      }
    }

    file.close ();
  }

  void
  ThreadRecognition::run ()
  {
    bool loop = true;
    while (loop)
    {
      loop = false;
      // pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_raw (new pcl::PointCloud<pcl::PointXYZRGB> ());
      // mutex_->lock ();
      // kinect2_grabber_->getPointCloud (scene_raw);
      // mutex_->unlock ();
      // scene_raw->sensor_orientation_.w() = 0.0;
      // scene_raw->sensor_orientation_.x() = 1.0;
      // scene_raw->sensor_orientation_.y() = 0.0;
      // scene_raw->sensor_orientation_.z() = 0.0;

      emit updateLogWindow ("[STATUS] Start recognition.");

      std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cluster_list;
      Segmentor segmentor;
      // QString file_name = "./Models/scene_1.pcd";
      QString file_name = "./Models/scene_home_4.pcd";
      // QString file_name = "./Models/table_tof_noise_noisy00000.pcd";
      // QString file_name = "./Models/table_tof_noise00000.pcd";
      segmentor.setRawPointCloud(file_name);
      segmentor.segment(cluster_list);

      emit updateLogWindow ("[INFO] Obtain " + QString::number(cluster_list.size()) + " clusters from the scene point cloud.");

      // Feature detection.
      for (int i = 0; i < cluster_list.size(); ++i)
      {
        emit updateVTKSegment (cluster_list[i]);
        /*
        emit updateLogWindow("[INFO] The " + QString::number(i) + "-th cluster has " + QString::number(cluster_list[i]->size()) + " points.");

        // Fetch feature points.
        radi::FeaturePointExtractor feature_point_extractor;
        feature_point_extractor.setInputCloud (cluster_list[i]);
        feature_point_extractor.compute ();

        std::vector<int> corner_indices = feature_point_extractor.getCornerPoints ();
        std::vector<int> edge_indices = feature_point_extractor.getEdgePoints ();
        std::vector<int> board_indices = feature_point_extractor.getBoardPoints ();
        std::vector<int> all_feature_indices = feature_point_extractor.getAllFeaturePoints ();

        emit updateVTKSegment (cluster_list[i], edge_indices, board_indices);
        // emit showFeaturePointsInVTKSegment(all_feature_indices);

        QString message = "[INFO] Obtain " + QString::number (corner_indices.size ()) +
            " corner points from current cluster.";
        emit updateLogWindow (message);
        message = "[INFO] Obtain " + QString::number (edge_indices.size ()) +
            " edge points from current cluster.";
        emit updateLogWindow (message);
        message = "[INFO] Obtain " + QString::number (board_indices.size ()) +
            " board points from current cluster.";
        emit updateLogWindow (message);

        radi::CEFEstimator cef_estimator;
        std::vector<radi::CornerEdgesFeature> corner_edges_feature_list;
        cef_estimator.setInputCloud (cluster_list[i]);
        cef_estimator.setFeaturePointIndices (all_feature_indices);
        cef_estimator.setCornerIndices (corner_indices);
        cef_estimator.esimate (corner_edges_feature_list);
        message = "[INFO] Obtain " + QString::number (corner_edges_feature_list.size ()) +
            " Corner Edges Features from current cluster.";
        emit updateLogWindow (message);

        radi::CNFEstimator cnf_estimator;
        std::vector<radi::CircleNormalFeature> circle_normal_feature_list;
        cnf_estimator.setInputCloud (cluster_list[i]);
        cnf_estimator.setFeaturePointIndices (all_feature_indices);
        cnf_estimator.esimate (circle_normal_feature_list);
        message = "[INFO] Obtain " + QString::number (circle_normal_feature_list.size ()) +
            " Circle Normal Features from current cluster.";
        emit updateLogWindow (message);

        // emit updateVTKSegment(point_cluster, cnf_estimator.);

        // Correspondance pair.
        radi::LGFCorrespGroup lgf_corresp_group;
        lgf_corresp_group.setSceneCloud (cluster_list[i]);
        lgf_corresp_group.setSceneFeatures(&corner_edges_feature_list, &circle_normal_feature_list);
        lgf_corresp_group.setFeatureIndices (boost::make_shared<std::vector<int> >(all_feature_indices));

        std::cout << "Features in scene point cloud: " << std::endl;
        std::cout << "corner-edge feature: " << corner_edges_feature_list.size() << std::endl;
        std::cout << "circle-normal feature: " << circle_normal_feature_list.size() << std::endl;
        // Match point cloud to each object in the model.
        Eigen::Vector3i color_cluster = cluster_list[i]->points[0].getRGBVector3i ();
        message = "[INFO] Color of current cluster: red " + QString::number (color_cluster[0])
            + ", green " + QString::number (color_cluster[1]) + ", blue " + QString::number (color_cluster[2]);
        emit updateLogWindow (message);
        QList<QString> keys = map_color_.keys();
        for (int i = 0; i < keys.size (); ++i)
        {
          timer_->start();

          message = "[STATUS] Matching object '" + keys[i] + "'.";
          emit updateLogWindow (message);

          lgf_corresp_group.setReferenceCloud (map_point_cloud_[keys[i]]);
          lgf_corresp_group.setReferenceFeatures (&(map_corner_[keys[i]]), &(map_circle_[keys[i]]));

          Eigen::Matrix4f mat_transform = Eigen::MatrixXf::Identity(4,4);
          if (lgf_corresp_group.recognize (mat_transform))
          {
            message = "[INFO] Successfully recognize the point cloud.";
            emit updateLogWindow (message);
          }
          else
          {
            message = "[INFO] Fail to recognize the point cloud.";
            emit updateLogWindow (message);
          }

          std::cout << "Time for object recognition in ms: " << timer_->elapsed() << std::endl;
        }

        */
        sleep (5);
      }

      emit updateLogWindow ("[STATUS] Finish recognition.");
    }
  }

  void
  ThreadRecognition::receivePointClouds (QMap<QString, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> map_point_cloud)
  {
    std::cout << "Receive point clouds." << std::endl;
    map_point_cloud_ = map_point_cloud;
  }

} // namespace radi
