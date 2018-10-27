#include <QFile>
#include <QXmlStreamReader>
#include <vtkRenderWindow.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/transforms.h>
#include "vtk_model.h"

namespace radi
{
  Eigen::Matrix3f
  rpy2matrix (const Eigen::Vector3f & rpy);

  VTKModel::VTKModel () : viewer_(new pcl::visualization::PCLVisualizer ("Object", false)),
      file_model_("")
  {
    viewer_->setBackgroundColor (0, 0, 0);
    this->SetRenderWindow (viewer_->getRenderWindow ());
    viewer_->setupInteractor (this->GetInteractor (), this->GetRenderWindow ());
    this->update ();
  }

  VTKModel::~VTKModel ()
  { }

  void
  VTKModel::setModelFile (const QString & file_model)
  {
    file_model_ = file_model;
    QFile file(file_model_);
    if (!file.open(QFile::ReadOnly|QFile::Text))
    {
      emit updateLogWindow ("[Error] " + QString("File '%1' cannot be found!").arg(file_model_));
      return;
    }

    QXmlStreamReader xml_reader;
    xml_reader.setDevice(&file);
    while (xml_reader.readNextStartElement())
    {
      if(xml_reader.name() == "scene")     // Enter element 'scene'.
      {
        while (xml_reader.readNextStartElement())
        {
          if (xml_reader.name() == "object")
          {
            QString object_name = xml_reader.attributes().value("name").toString();
            this->object_names_.append (object_name);

            emit updateLogWindow("[INFO] Find object '" + object_name + "'.");

            while (xml_reader.readNextStartElement())
            {
              if (xml_reader.name() == "visual")
              {
                while (xml_reader.readNextStartElement ())
                {
                  if (xml_reader.name() == "origin")
                  {
                    QString rpy = xml_reader.attributes().value("rpy").toString();
                    QVector<QString> rpy_list = rpy.split(" ", QString::SkipEmptyParts).toVector();
                    Eigen::Vector3f vect_rpy = Eigen::Vector3f(rpy_list[0].toFloat(),
                        rpy_list[1].toFloat(), rpy_list[2].toFloat());
                    QString xyz = xml_reader.attributes().value("xyz").toString();
                    QVector<QString> xyz_list = xyz.split(" ", QString::SkipEmptyParts).toVector();
                    Eigen::Matrix4f mat_frame = Eigen::MatrixXf::Identity(4,4);
                    mat_frame.block(0,0,3,3) = rpy2matrix (vect_rpy);
                    mat_frame.block(0,3,3,1) = Eigen::Vector3f(xyz_list[0].toFloat(),
                        xyz_list[1].toFloat(), xyz_list[2].toFloat());
                    this->map_frame_[object_name] = mat_frame;

                    xml_reader.readElementText();
                  }
                  else if (xml_reader.name() == "geometry")
                  {
                    while (xml_reader.readNextStartElement ())
                    {
                      if (xml_reader.name() == "mesh")
                      {
                        QString file_name = xml_reader.attributes().value("filename").toString();
                        this->map_files_[object_name] = file_name;
                        xml_reader.readElementText();
                      }
                    }
                  }
                  else if (xml_reader.name() == "material")
                  {
                    while (xml_reader.readNextStartElement ())
                    {
                      if (xml_reader.name() == "color")
                      {
                        QString color = xml_reader.attributes().value("rgb").toString();
                        QVector<QString> color_list = color.split(" ", QString::SkipEmptyParts).toVector();
                        this->map_color_[object_name] = QColor(color_list[0].toInt(),
                            color_list[1].toInt(), color_list[2].toInt());
                        xml_reader.readElementText();
                      }
                    }
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
      }
    }

    file.close ();

    QMap<QString, Eigen::Matrix4f> map_transform;
    for (int i = 0; i < object_names_.size(); ++i)
    {
      pcl::PolygonMesh mesh;
      pcl::io::loadPolygonFileSTL(map_files_[object_names_[i]].toStdString(), mesh);
      map_polygon_mesh_[object_names_[i]] = mesh;

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
      pcl::fromPCLPointCloud2 (map_polygon_mesh_[object_names_[i]].cloud, *point_cloud);
      map_point_cloud_[object_names_[i]] = point_cloud;

      map_in_viewer_[object_names_[i]] = false;

      map_transform[object_names_[i]] = Eigen::MatrixXf::Identity(4,4);
    }

    emit sendPointClouds (map_point_cloud_);
    updateModel (map_transform);
  }

  void
  VTKModel::updateModel (QMap<QString, Eigen::Matrix4f> map_transform)
  {
    QList<QString> keys = map_transform.keys();
    for (int i = 0; i < keys.size(); ++i)
    {
      if (object_names_.contains (keys[i]))
      {
        pcl::PolygonMesh mesh = map_polygon_mesh_[keys[i]];
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromPCLPointCloud2 (mesh.cloud, cloud);
        pcl::transformPointCloud (cloud, cloud, map_transform[keys[i]]);
        pcl::toPCLPointCloud2 (cloud, mesh.cloud);

        if (map_in_viewer_[keys[i]])
        {
          viewer_->updatePolygonMesh(mesh, keys[i].toStdString());
        }
        else
        {
          map_in_viewer_[keys[i]] = true;
          viewer_->addPolygonMesh(mesh, keys[i].toStdString());
          viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
              static_cast<double>(map_color_[keys[i]].red()/255.0),
              static_cast<double>(map_color_[keys[i]].green()/255.0),
              static_cast<double>(map_color_[keys[i]].blue()/255.0), keys[i].toStdString());
        }
      }
    }

    for (int i = 0; i < object_names_.size(); ++i)
    {
      if (!keys.contains(object_names_[i]))
      {
        if (map_in_viewer_[object_names_[i]])
        {
          viewer_->removePolygonMesh(object_names_[i].toStdString());
          map_in_viewer_[object_names_[i]] = false;
        }
      }
    }

    this->update ();
  }

  Eigen::Matrix3f
  rpy2matrix(const Eigen::Vector3f &rpy)
  {
    Eigen::Matrix3f mat_rotation;
    mat_rotation(0,0) = std::cos(rpy[2]) * std::cos(rpy[1]);
    mat_rotation(1,0) = std::sin(rpy[2]) * std::cos(rpy[1]);
    mat_rotation(2,0) = -std::sin(rpy[1]);

    mat_rotation(0,1) = std::cos(rpy[2])*std::sin(rpy[1])*std::sin(rpy[0]) - std::sin(rpy[2])*std::cos(rpy[0]);
    mat_rotation(1,1) = std::sin(rpy[2])*std::sin(rpy[1])*std::sin(rpy[0]) + std::cos(rpy[2])*std::cos(rpy[0]);
    mat_rotation(2,1) = std::cos(rpy[1]) * std::sin(rpy[0]);

    mat_rotation(0,2) = std::cos(rpy[2])*std::sin(rpy[1])*std::cos(rpy[0]) + std::sin(rpy[2])*std::sin(rpy[0]);
    mat_rotation(1,2) = std::sin(rpy[2])*std::sin(rpy[1])*std::cos(rpy[0]) - std::cos(rpy[2])*std::sin(rpy[0]);
    mat_rotation(2,2) = std::cos(rpy[1]) * std::cos(rpy[0]);

    return (mat_rotation);
  }

} // namespace radi
