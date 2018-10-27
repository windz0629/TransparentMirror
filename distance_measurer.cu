#include <math.h>   // CUDA Math.
#include <thrust/sort.h>
#include <thrust/functional.h>

#include "distance_measurer.h"

namespace radi {

  DistanceMeasurer::DistanceMeasurer () : num_points_(0)
  { }

  DistanceMeasurer::~DistanceMeasurer ()
  {
    if (dev_num_points_)
      cudaFree(dev_num_points_);

    if (dev_points_)
      cudaFree(dev_points_);
  }

  void
  DistanceMeasurer::setReferPointCloud (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr refer_point_cloud)
  {
    num_points_ = refer_point_cloud->size ();
    cudaMalloc ((void **)&dev_num_points_, sizeof(int));
    cudaMemcpy(dev_num_points_, &num_points_, sizeof(int), cudaMemcpyHostToDevice);

    int total_size = num_points_ * 3;
    float * host_points = (float *)malloc (total_size*sizeof(float));
    for (int i = 0; i < num_points_; ++i)
    {
      host_points[i*3+0] = refer_point_cloud->points[i].x;
      host_points[i*3+1] = refer_point_cloud->points[i].y;
      host_points[i*3+2] = refer_point_cloud->points[i].z;
    }

    cudaMalloc (&dev_points_, total_size*sizeof(float));
    cudaMemcpy (dev_points_, host_points, total_size*sizeof(float), cudaMemcpyHostToDevice);

    free (host_points);
  }

  float
  DistanceMeasurer::calShortestDistance (const pcl::PointXYZRGB & point)
  {
    float host_point[3];
    host_point[0] = point.x;
    host_point[1] = point.y;
    host_point[2] = point.z;

    float * dev_point;
    cudaMalloc ((void **)&dev_point, 3*sizeof(float));
    cudaMemcpy(dev_point, host_point, 3*sizeof(float), cudaMemcpyHostToDevice);

    float * dev_distances;
    cudaMalloc ((void **)&dev_distances, num_points_*sizeof(float));
    distPoint2Point<<<(num_points_+255)/256, 256>>> (dev_point, dev_points_, dev_num_points_, dev_distances);

    float * distances = (float *) malloc (num_points_*sizeof(float));
    cudaMemcpy(distances, dev_distances, num_points_*sizeof(float), cudaMemcpyDeviceToHost);

    thrust::stable_sort (distances, distances+num_points_, thrust::less_equal<float> ());

    float min_distance = distances[0];

    free (distances);
    cudaFree (dev_point);
    cudaFree (dev_distances);

    return (min_distance);
  }

  __global__ void
  distPoint2Point (const float * dev_point, const float * dev_points, const int * dev_num_points, float * dev_distances)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < dev_num_points[0])
    {
      const float * dev_point_in_cloud = &dev_points[tid*3];
      dev_distances[tid] = norm3df (dev_point[0]-dev_point_in_cloud[0],
          dev_point[1]-dev_point_in_cloud[1], dev_point[2]-dev_point_in_cloud[2]);
    }
  }
}
