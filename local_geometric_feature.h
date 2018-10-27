/*
 * Define Local Geometric Features as well as the related classes.
 *
 */

#ifndef RADI_LOCAL_GEOMETRIC_FEATURE_H_
#define RADI_LOCAL_GEOMETRIC_FEATURE_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

namespace radi
{

  /*!
   * \class CornerEdgesFeature
   * \brief CornerEdgesFeature利用角点和由角点延伸的边构成特征，至少需要2条边。边的分布是有序的，因此它们的夹角也是一组序列。利用这组
   * 夹角序列对点云中的特征和三维模型中指定的特征进行匹配，进而对相对位姿进行初估。
   * \details CornerEdgesFeature利用角点和由角点延伸的边构成特征，至少需要2条边。边的分布是有序的，因此它们的夹角也是一组序列。利用这组
   * 夹角序列对点云中的特征和三维模型中指定的特征进行匹配，进而对相对位姿进行初估。。匹配了后，由角点的位置计算相对位置，由边计算相对姿态。
   * 可能产生多个候选相对位姿；此时，可以利用其他Local Geometric Feature进行初步排除；进一步，变换点云，跟模型进行比较，合适的候选相对位姿
   * 会使得变换后的点云与模型是接近的，而错误的候选相对位姿会导致变换后的点云与模型有较大的差距。候选相对位姿的数量不会太大；对于每个初步排除的
   * 候选相对位姿，与模型进行一次比对即可验证相对位姿的有效性。
   */
  class CornerEdgesFeature
  {
    public:
      CornerEdgesFeature (pcl::PointXYZ corner = pcl::PointXYZ());
      CornerEdgesFeature (pcl::PointXYZ corner, std::vector<Eigen::Vector3f> edges);
      ~CornerEdgesFeature ();

      template<typename PointType> void
      setCorner (PointType corner)
      {
        corner_ = pcl::PointXYZ(corner.x, corner.y, corner.z);
      }

      void
      setCorner (const Eigen::Vector3f & position)
      {
        corner_ = pcl::PointXYZ(position[0], position[1], position[2]);
      }

      void
      setEdges (const std::vector<Eigen::Vector3f> & edges);

      void
      appendEdge (const Eigen::Vector3f & edge);

      /*!
       * \fn void compute ()
       * \brief 计算夹角序列。
       */
      void
      compute ();

      const pcl::PointXYZ
      getCorner();

      const Eigen::Vector3f
      getCornerPosition() const;

      const Eigen::Vector3f
      getEdge (std::size_t index) const;

      const std::vector<Eigen::Vector3f> &
      getEdges () const;

      const std::vector<float> &
      getIncludedAngleSequence () const;

      const std::vector<std::vector<int> > &
      getEdgePairSequence () const;

      std::size_t
      getNumEdges () const;

    private:
      pcl::PointXYZ corner_;
      std::vector<Eigen::Vector3f> edges_;
      std::vector<float> included_angle_sequence_;    // 边与边夹角的序列。
      std::vector<std::vector<int> > edge_pair_sequence_;   // 对应夹角序列的边的索引对。
  };

  // Refine CVS feature list, for example, remove extra features which are two close with each other or remove features
  // which have smaller edge number than the threshold.
  void
  refineCEFeatures (const std::vector<CornerEdgesFeature> & source_features, std::vector<CornerEdgesFeature> & target_features);

  void
  transformCEFeature (const Eigen::Matrix4f & mat_transf,
          const CornerEdgesFeature & source_feature, CornerEdgesFeature & target_feature);

  bool
  isInList(int index, std::vector<int> index_list);

  /*!
   * \class CircleNormalFeature
   * \brief CircleNormalFeature利用圆形边界的圆心及其法线方向构成特征。利用圆的半径对特征进行匹配。将模型或点云向圆形边界所在的平面
   * 进行投影，分布的主轴方向也是一个辅助参数。
   * \details CircleNormalFeature利用圆形边界的圆心及其法线方向构成特征。利用圆的半径对特征进行匹配。将模型或点云向圆形边界所在的平面
   * 进行投影，分布的主轴方向也是一个辅助参数。如果没有主轴方向这个辅助参数，则可以以固定间距绕圆形边界的法线方向转动，得到一组候选相对位姿；
   * 此时，变换点云，跟模型进行比较，合适的候选相对位姿会使得变换后的点云与模型是接近的，而错误的候选相对位姿会导致变换后的点云与模型有较大的
   * 差距。如果所有候选位姿都使得点云与模型比较接近，选取目标值最大的那个。
   *
   * 主轴只用来估计绕法线的转动角，即优先保证CircleNormalFeature的法线是共线的，引入主轴是为了确定绕法线的偏转角。
   */
  class CircleNormalFeature
  {
    public:
      CircleNormalFeature ();
      ~CircleNormalFeature ();

      void
      setCenter (const Eigen::Vector3f & center);

      void
      setNormal (const Eigen::Vector3f & normal);

      void
      setRadius (float radius);

      void
      setPrincipalAxes (const std::vector<Eigen::Vector3f> & principal_axes);

      void
      appendPrincipalAxis (const Eigen::Vector3f & principal_axis);

      /*!
       * \fn void compute ()
       * \brief 计算夹角序列。
       */
      void
      compute ();

      const Eigen::Vector3f &
      getCenter () const;

      const Eigen::Vector3f &
      getNormal () const;

      float
      getRadius () const;

      const Eigen::Vector3f
      getPrincipalAxis (std::size_t index) const;

      const std::vector<Eigen::Vector3f> &
      getPrincipalAxes () const;

      const std::vector<float> &
      getIncludedAngleSequence () const;

      const std::vector<std::vector<int> > &
      getPrincipalAxisPairSequence () const;

      std::size_t
      getNumPrincipalAxes () const;

    private:
      Eigen::Vector3f center_;
      Eigen::Vector3f normal_;
      float radius_;
      std::vector<Eigen::Vector3f> principal_axes_;
      std::vector<float> included_angle_sequence_;    // 主轴之间夹角的序列。
      std::vector<std::vector<int> > principal_axis_pair_sequence_;   // 对应的主轴的索引对。
  };

  void
  transformCNFeature (const Eigen::Matrix4f & mat_transf, const CircleNormalFeature & source_feature, CircleNormalFeature & target_feature);

} // namespace radi

#endif // RADI_LOCAL_GEOMETRIC_FEATURE_H_
