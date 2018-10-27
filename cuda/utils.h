/*
 * Extract the corner points, points on edges and creases.
 *
 */

#ifndef RADI_UTILS_H_
#define RADI_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>

extern __constant__ float SRL_EPS;
extern __constant__ float SRL_PI;

__device__ float srl_sign (float value);

__device__ float srl_sqrt(float scalar);

__device__ float srl_asin(float scalar);

__device__ float srl_acos(float scalar);

__device__ void srl_svd (const float * matrix_a, int num_dim, float * matrix_u, float * matrix_s, float * matrix_v, float tolerance);

__device__ void srl_matrix_identity (float * mat_a, int num_row, int num_column);

__device__ void srl_matrix_zeros (float * mat_a, int num_row, int num_column);

__device__ void srl_matrix_ones (float * mat_a, int num_row, int num_column);

__device__ void srl_matrix_copy (const float * mat_src, int length , float * mat_dst);

__device__ void srl_matrix_add (const float * mat_a, const float * mat_b, int num_row, int num_column, float * mat_c);

__device__ void srl_matrix_subtract (const float * mat_a, const float * mat_b, int num_row, int num_column, float * mat_c);

__device__ void srl_matrix_multiply (const float * mat_a, const float * mat_b, int num_row_a, int num_mid, int num_col_b, float * mat_c);

__device__ void srl_matrix_multiplied_by_scalar (const float * mat_a, int num_row, int num_column, float scalar, float * mat_b);

__device__ void srl_matrix_divided_by_scalar (const float * mat_a, int num_row, int num_column, float scalar, float * mat_b);

__device__ void srl_matrix_transpose (const float * mat_a, int num_row, int num_column, float * mat_t);

__device__ float srl_dot_product (const float * mat_a, const float * mat_b, int length);

__device__ void srl_cross_product (const float * vect_a, const float * vect_b, float * vect_c);

__device__ void srl_exp_map_se(const float * xi, bool with_rotation_angle, float * mat_transform, float angle);

__device__ void srl_exp_map_so(const float * omega, bool with_rotation_angle, float * mat_transform, float angle);

#endif // RADI_FEATURE_EXTRACTOR_H_
