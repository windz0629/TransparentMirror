#include <boost/make_shared.hpp>
#include <Eigen/Eigenvalues>

#include <pcl/octree/octree_search.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/filters/extract_indices.h>

#include "local_geometric_feature.h"

namespace radi
{
  CornerEdgesFeature::CornerEdgesFeature (pcl::PointXYZ corner)
          : corner_(corner), edges_(std::vector<Eigen::Vector3f>()),
            included_angle_sequence_(std::vector<float>()),
            edge_pair_sequence_(std::vector<std::vector<int> >())
  { }

  CornerEdgesFeature::CornerEdgesFeature (pcl::PointXYZ corner, std::vector<Eigen::Vector3f> edges)
          : corner_(corner), edges_(edges),
            included_angle_sequence_(std::vector<float>()),
            edge_pair_sequence_(std::vector<std::vector<int> >())
  { }

  CornerEdgesFeature::~CornerEdgesFeature ()
  { }

  void
  CornerEdgesFeature::setEdges (const std::vector<Eigen::Vector3f> & edges)
  {
    edges_ = edges;
  }

  void
  CornerEdgesFeature::appendEdge (const Eigen::Vector3f & edge)
  {
    edges_.push_back(edge);
  }

  void
  CornerEdgesFeature::compute ()
  {
    int nCountTotal;
    if (edges_.size() == 2)
      nCountTotal = 1;
    else
      nCountTotal = edges_.size();

    Eigen::Vector3f center_vector = Eigen::Matrix3Xf::Zero(3,1);
    for (std::size_t i = 0; i < edges_.size(); ++i)
    {
      center_vector += edges_[i];
    }
    center_vector /= std::sqrt(center_vector.dot(center_vector));

    std::vector<Eigen::Vector3f> normal_list(edges_.size());
    for (std::size_t i = 0; i < edges_.size(); ++i)
    {
      normal_list[i] = edges_[i].cross(center_vector);
      normal_list[i] /= std::sqrt(normal_list[i].dot(normal_list[i]));
    }

    std::vector<float> angle_plane_list(edges_.size());
    angle_plane_list[0] = 0.0;
    for (std::size_t i = 1; i < edges_.size(); ++i)
    {
      float vector_included_angle = std::acos(normal_list[0].dot(normal_list[i]));
      if ((normal_list[0].cross(normal_list[i])).dot(center_vector) > 0.0)
      {
        angle_plane_list[i] = 2*3.1415926 - vector_included_angle;
      }
      else
      {
        angle_plane_list[i] = vector_included_angle;
      }
    }

    // Sort.
    std::vector<std::size_t> order_indices(angle_plane_list.size());
    std::iota(order_indices.begin (), order_indices.end (), 0);
    std::sort(order_indices.begin(), order_indices.end(), [&angle_plane_list](int idx_1, int idx_2)
        { return angle_plane_list[idx_1] < angle_plane_list[idx_2]; });

    for (std::size_t i = 0; i < order_indices.size()-1; ++i)
    {
      std::vector<int> pair_indicies(2);
      pair_indicies[0] = order_indices[i];
      pair_indicies[1] = order_indices[i+1];
      included_angle_sequence_.push_back(std::acos(edges_[pair_indicies[0]].dot(edges_[pair_indicies[1]])));
      edge_pair_sequence_.push_back(pair_indicies);
    }

    if (nCountTotal > 1)
    {
      std::vector<int> pair_indicies(2);
      pair_indicies[0] = order_indices[nCountTotal-1];
      pair_indicies[1] = order_indices[0];
      included_angle_sequence_.push_back(std::acos(edges_[pair_indicies[0]].dot(edges_[pair_indicies[1]])));
      edge_pair_sequence_.push_back(pair_indicies);
    }
  }

  const pcl::PointXYZ
  CornerEdgesFeature::getCorner ()
  {
    return (corner_);
  }

  const Eigen::Vector3f
  CornerEdgesFeature::getCornerPosition () const
  {
    return (Eigen::Vector3f(corner_.x, corner_.y, corner_.z));
  }

  const Eigen::Vector3f
  CornerEdgesFeature::getEdge (std::size_t index) const
  {
    return (edges_[index]);
  }

  const std::vector<Eigen::Vector3f> &
  CornerEdgesFeature::getEdges () const
  {
    return (edges_);
  }

  const std::vector<float> &
  CornerEdgesFeature::getIncludedAngleSequence () const
  {
    return (included_angle_sequence_);
  }

  const std::vector<std::vector<int> > &
  CornerEdgesFeature::getEdgePairSequence () const
  {
    return (edge_pair_sequence_);
  }

  std::size_t
  CornerEdgesFeature::getNumEdges() const
  {
    return (edges_.size());
  }

  void
  refineCEFeatures (const std::vector<CornerEdgesFeature> & source_features, std::vector<CornerEdgesFeature> & target_features)
  {
    // ToDo:
  }

  void
  transformCEFeature (const Eigen::Matrix4f & mat_transf,
          const CornerEdgesFeature & source_feature, CornerEdgesFeature & target_feature)
  {
    Eigen::Vector3f pos_corner = source_feature.getCornerPosition();
    Eigen::Vector3f pos_transformed = mat_transf.block(0,0,3,3)*pos_corner + mat_transf.block(0,3,3,1);
    target_feature.setCorner(pcl::PointXYZ(pos_transformed[0], pos_transformed[1], pos_transformed[2]));

    std::vector<Eigen::Vector3f> edges (source_feature.getNumEdges ());
    for (std::size_t i = 0; i < source_feature.getNumEdges(); ++i)
    {
      edges[i] = mat_transf.block(0,0,3,3) * source_feature.getEdge (i);
    }
    target_feature.setEdges (edges);

    target_feature.compute ();
  }

  bool
  isInList(int index, std::vector<int> index_list)
  {
    if (index_list.empty())
    {
      return (true);
    }
    else
    {
      for (std::size_t i = 0; i < index_list.size(); ++i)
      {
        if (index == index_list[i])
          return (true);
      }

      return (false);
    }
  }

  CircleNormalFeature::CircleNormalFeature () : center_(), normal_()
  { }

  CircleNormalFeature::~CircleNormalFeature ()
  { }

  void
  CircleNormalFeature::setCenter (const Eigen::Vector3f & center)
  {
    center_ = center;
  }

  void
  CircleNormalFeature::setNormal (const Eigen::Vector3f & normal)
  {
    normal_ = normal;
  }

  void
  CircleNormalFeature::setRadius (float radius)
  {
    radius_ = radius;
  }

  const Eigen::Vector3f &
  CircleNormalFeature::getCenter () const
  {
    return (center_);
  }

  void
  CircleNormalFeature::setPrincipalAxes (const std::vector<Eigen::Vector3f> & principal_axes)
  {
    principal_axes_ = principal_axes;
  }

  void
  CircleNormalFeature::appendPrincipalAxis (const Eigen::Vector3f & principal_axis)
  {
    principal_axes_.push_back(principal_axis);
  }

  void
  CircleNormalFeature::compute ()
  {
    if (principal_axes_.size() > 0)
    {
      // 以第一个主轴为基准，计算它与其他主轴的夹角。
      Eigen::Vector3f reference = principal_axes_[0];
      std::vector<float> included_angle_list (principal_axes_.size ());
      for (std::size_t i = 0; i < principal_axes_.size(); ++i)
      {
        float included_angle = std::acos(reference.dot(principal_axes_[i]));
        if ((reference.cross(principal_axes_[i])).dot(normal_) <= 0.0)
          included_angle_list[i] = 2*3.1415926 - included_angle;
        else
          included_angle_list[i] = included_angle;
      }

      // 对夹角进行排序，得到夹角序列和对应的主轴的index。
      std::vector<std::size_t> order_indices (included_angle_list.size ());
      std::iota(order_indices.begin (), order_indices.end (), 0);
      std::sort(order_indices.begin(), order_indices.end(), [&included_angle_list](int idx_1, int idx_2)
          { return included_angle_list[idx_1] < included_angle_list[idx_2]; });

      if (included_angle_list.size () == 1)
      {
        // ToDo: 只有一个主轴的话，夹角序列应该为空。
        included_angle_sequence_.push_back (included_angle_list[0]);
        std::vector<int> pair_indicies(2);
        pair_indicies[0] = 0;
        pair_indicies[1] = 0;
        principal_axis_pair_sequence_.push_back (pair_indicies);
      }
      else
      {
        for (std::size_t i = 0; i < order_indices.size()-1; ++i)
        {
          included_angle_sequence_.push_back (included_angle_list[i+1] - included_angle_list[i]);
          std::vector<int> pair_indicies(2);
          pair_indicies[0] = order_indices[i];
          pair_indicies[1] = order_indices[i+1];
          principal_axis_pair_sequence_.push_back (pair_indicies);
        }
      }
    }
  }

  const Eigen::Vector3f &
  CircleNormalFeature::getNormal () const
  {
    return (normal_);
  }

  float
  CircleNormalFeature::getRadius () const
  {
    return (radius_);
  }

  const Eigen::Vector3f
  CircleNormalFeature::getPrincipalAxis (std::size_t index) const
  {
    return (principal_axes_[index]);
  }

  const std::vector<Eigen::Vector3f> &
  CircleNormalFeature::getPrincipalAxes () const
  {
    return (principal_axes_);
  }

  const std::vector<float> &
  CircleNormalFeature::getIncludedAngleSequence () const
  {
    return (included_angle_sequence_);
  }

  const std::vector<std::vector<int> > &
  CircleNormalFeature::getPrincipalAxisPairSequence() const
  {
    return (principal_axis_pair_sequence_);
  }

  std::size_t
  CircleNormalFeature::getNumPrincipalAxes () const
  {
    return (principal_axes_.size ());
  }

  void
  transformCNFeature (const Eigen::Matrix4f & mat_transf, const CircleNormalFeature & source_feature, CircleNormalFeature & target_feature)
  {
    const Eigen::Vector3f & center = source_feature.getCenter ();
    const Eigen::Vector3f & normal = source_feature.getNormal ();
    Eigen::Vector3f center_transformed = mat_transf.block (0,0,3,3)*center + mat_transf.block (0,3,3,1);
    Eigen::Vector3f normal_transformed = mat_transf.block (0,0,3,3)*normal;
    target_feature.setCenter (center_transformed);
    target_feature.setNormal (normal_transformed);

    target_feature.setRadius (source_feature.getRadius ());

    std::vector<Eigen::Vector3f> principal_axes (source_feature.getNumPrincipalAxes ());
    for (std::size_t i = 0; i < source_feature.getNumPrincipalAxes (); ++i)
    {
      principal_axes[i] = mat_transf.block(0,0,3,3) * source_feature.getPrincipalAxis (i);
    }
    target_feature.setPrincipalAxes (principal_axes);

    target_feature.compute();
  }

} // namespace radi
