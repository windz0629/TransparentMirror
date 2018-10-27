#include <math.h>
#include "utils.h"

__device__ void bidiagonalize(const float * matrix_a, int num_dim, float * matrix_u, float * matrix_b, float * matrix_v);
__device__ void svd_bidiagonal_matrix(float * matrix_b, float * matrix_u, float * matrix_v, int num_dim, float tolerance);
__device__ void golub_kahan_svd_step(float * matrix_b, float * matrix_p, float * matrix_q, int num_dim);

__constant__ float SRL_EPS = 1E-7;
__constant__ float SRL_PI = 3.1415927;

__device__ float srl_sign(float value)
{
    if (value >= 0.0)
        return (1.0);
    else
        return (-1.0);
}

__device__ float srl_sqrt(float scalar)
{
    if ((scalar > -SRL_EPS) && (scalar < 0.0))
        scalar = 0.0;

    return (sqrtf(scalar));
}

__device__ float srl_asin(float scalar)
{
    if ((scalar > 1.0) && (scalar < (1.0+SRL_EPS)))
        scalar = 1.0;
    else if ((scalar < -1.0) && (scalar > (-1.0-SRL_EPS)))
        scalar = -1.0;

    return (asinf(scalar));
}

__device__ float srl_acos(float scalar)
{
    if ((scalar > 1.0) && (scalar < (1.0+SRL_EPS)))
        scalar = 1.0;
    else if ((scalar < -1.0) && (scalar > (-1.0-SRL_EPS)))
        scalar = -1.0;

    return (acosf(scalar));
}

__device__ void srl_svd (const float * matrix_a, int num_dim, float * matrix_u, float * matrix_s, float * matrix_v, float tolerance)
{
    float * mat_u = (float *)malloc((num_dim*num_dim)*sizeof(float));
    float * mat_v = (float *)malloc((num_dim*num_dim)*sizeof(float));
    float * mat_b = (float *)malloc((num_dim*num_dim)*sizeof(float));
    bidiagonalize(matrix_a, num_dim, mat_u, mat_b, mat_v);

    float * mat_p = (float *)malloc((num_dim*num_dim)*sizeof(float));
    float * mat_q = (float *)malloc((num_dim*num_dim)*sizeof(float));
    svd_bidiagonal_matrix(mat_b, mat_p, mat_q, num_dim, tolerance);

    srl_matrix_multiply(mat_u, mat_p, num_dim, num_dim, num_dim, matrix_u);
    srl_matrix_multiply(mat_v, mat_q, num_dim, num_dim, num_dim, matrix_v);
    for (int i = 0; i < (num_dim*num_dim); ++i)
    {
        matrix_s[i] = mat_b[i];
    }

    free(mat_u);
    free(mat_v);
    free(mat_b);
    free(mat_p);
    free(mat_q);
}

__device__ void bidiagonalize(const float * matrix_a, int num_dim, float * matrix_u, float * matrix_b, float * matrix_v)
{
    srl_matrix_copy(matrix_a, num_dim*num_dim, matrix_b);
    srl_matrix_identity(matrix_u, num_dim, num_dim);
    srl_matrix_identity(matrix_v, num_dim, num_dim);

    for (std::size_t k = 0; k < num_dim; ++k)
    {
        float * vect_householder = (float *)malloc(num_dim*sizeof(float));
        for (std::size_t i = 0; i < num_dim; ++i)
            vect_householder[i] = matrix_b[i+k*num_dim];

        for (std::size_t i = 0; i < k; ++i)
        {
            vect_householder[i] = 0.0;
        }
        vect_householder[k] = 1.0;

        float sigma = 0.0;
        if (k+1 < num_dim)
        {
            for (int i = k+1; i < num_dim; ++i)
            {
                sigma += matrix_b[i+k*num_dim] * matrix_b[i+k*num_dim];
            }
        }

        float beta;
        if (sigma < SRL_EPS && matrix_b[k+k*num_dim] >= 0)
        {
            beta = 0.0;
        }
        else if (sigma < SRL_EPS && matrix_b[k+k*num_dim] < 0)
        {
            beta = -2.0;
        }
        else
        {
            float mu = srl_sqrt(powf(matrix_b[k+k*num_dim],2) + sigma);
            float k_elem;
            if (matrix_b[k+k*num_dim] < 0)
            {
                k_elem = matrix_b[k+k*num_dim] - mu;
            }
            else
            {
                k_elem = -sigma / (matrix_b[k+k*num_dim] + mu);
            }

            beta = 2.0 * powf(k_elem, 2) / (sigma + powf(k_elem,2));
            vect_householder[k] = k_elem;
            for (int i = 0; i < num_dim; ++i)
                vect_householder[i] /= k_elem;
        }

        // Householder reflection matrix.
        float * mat_q = (float *)malloc((num_dim*num_dim)*sizeof(float));
        for (int i = 0; i < num_dim; ++i)
        {
            for (int j = 0; j < num_dim; ++j)
            {
                if (i == j)
                    mat_q[i+j*num_dim] = 1.0 - beta*vect_householder[i]*vect_householder[j];
                else
                    mat_q[i+j*num_dim] = 0.0 - beta*vect_householder[i]*vect_householder[j];
            }
        }

        float * matrix_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
        srl_matrix_multiply(mat_q, matrix_b, num_dim, num_dim, num_dim, matrix_temp);
        srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);

        srl_matrix_multiply(matrix_u, mat_q, num_dim, num_dim, num_dim, matrix_temp);
        srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_u);

        free(matrix_temp);
        free(vect_householder);

        if (k < num_dim-2)
        {
            float * vect_householder = (float *)malloc(num_dim*sizeof(float));
            for (std::size_t j = 0; j < num_dim; ++j)
                vect_householder[j] = matrix_b[k+j*num_dim];

            for (std::size_t i = 0; i < k+1; ++i)
            {
                vect_householder[i] = 0.0;
            }
            vect_householder[k+1] = 1.0;

            float sigma = 0.0;
            for (int i = k+2; i < num_dim; ++i)
                sigma += matrix_b[k+i*num_dim] * matrix_b[k+i*num_dim];

            float beta;
            if (sigma < SRL_EPS && matrix_b[k+(k+1)*num_dim] >= 0)
            {
                beta = 0.0;
            }
            else if (sigma < SRL_EPS && matrix_b[k+(k+1)*num_dim] < 0)
            {
                beta = -2.0;
            }
            else
            {
                float mu = srl_sqrt(powf(matrix_b[k+(k+1)*num_dim], 2) + sigma);
                float temp;   // (k+1)-th element;
                if (matrix_b[k+(k+1)*num_dim] < 0.0)
                {
                    temp = matrix_b[k+(k+1)*num_dim] - mu;
                }
                else
                {
                    temp = -sigma / (matrix_b[k+(k+1)*num_dim] + mu);
                }
                beta = 2.0 * powf(temp, 2) / (sigma + powf(temp, 2));
                vect_householder[k+1] = temp;
                for (int i = 0; i < num_dim; ++i)
                    vect_householder[i] /= temp;
            }

            // Householder reflection matrix.
            float * mat_p = (float *)malloc((num_dim*num_dim)*sizeof(float));
            for (int i = 0; i < num_dim; ++i)
            {
                for (int j = 0; j < num_dim; ++j)
                {
                    if (i == j)
                        mat_p[i+j*num_dim] = 1.0 - beta*vect_householder[i]*vect_householder[j];
                    else
                        mat_p[i+j*num_dim] = 0.0 - beta*vect_householder[i]*vect_householder[j];
                }
            }

            // matrixB = matrixB * matP;
            // matrixV = matrixV * matP;
            float * matrix_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
            srl_matrix_multiply(matrix_b, mat_p, num_dim, num_dim, num_dim, matrix_temp);
            srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);

            srl_matrix_multiply(matrix_v, mat_p, num_dim, num_dim, num_dim, matrix_temp);
            srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_v);

            free(matrix_temp);
            free(vect_householder);
        }
    }
}

__device__ void svd_bidiagonal_matrix(float * matrix_b, float * matrix_u, float * matrix_v, int num_dim, float tolerance)
{
    srl_matrix_identity(matrix_u, num_dim, num_dim);
    srl_matrix_identity(matrix_v, num_dim, num_dim);

    while (true)
    {
        for (std::size_t i = 0; i < num_dim-1; ++i)
        {
            // if (fabsf(matrix_b[i+(i+1)*num_dim]) <= tolerance*(fabsf(matrix_b[i+i*num_dim])+fabsf(matrix_b[i+1+(i+1)*num_dim])))
            if (fabsf(matrix_b[i+(i+1)*num_dim]) <= SRL_EPS)
            {
                matrix_b[i+(i+1)*num_dim] = 0.0;
            }
        }

        // Find the largest q and the smallest p.
        std::size_t q = 0;
        for (int i = num_dim-1; i >= 0; --i)
        {
            if (0 == i)
            {
                q++;
                break;
            }

            if (fabsf(matrix_b[(i-1)+i*num_dim]) < SRL_EPS)
            {
                q++;
            }
            else
            {
                break;
            }
        }

        if (q == num_dim)
        {
            break;
        }

        std::size_t dimB_2 = 0;
        for (int i = num_dim-q-1; i >=0 ; --i)
        {
            if (0 == i) {
                dimB_2++;
                break;
            }

            if (fabsf(matrix_b[(i-1)+i*num_dim]) > SRL_EPS)
            {
                dimB_2++;
            }
            else
            {
                dimB_2++;
                break;
            }
        }

        std::size_t p = num_dim - q - dimB_2;

        float * matB_2 = (float *)malloc((dimB_2*dimB_2)*sizeof(float));
        for (int i = 0; i < dimB_2; ++i)
            for (int j = 0; j < dimB_2; ++j)
                matB_2[i+j*dimB_2] = matrix_b[(p+i)+(p+j)*num_dim];

        bool flag_zero_diag = false;
        for (std::size_t i = 0; i < dimB_2-1; ++i)
        {
            if (fabsf(matB_2[i+i*dimB_2]) < SRL_EPS)
            {
                flag_zero_diag = true;
                // Zero the superdiagonal entry in the same row.
                // Construct Givens rotation matrix.
                float * givens_matrix = (float *)malloc((num_dim*num_dim)*sizeof(float));
                srl_matrix_identity(givens_matrix, num_dim, num_dim);

                float c = 0.0;
                float s = -srl_sign(matB_2[i+(i+1)*dimB_2]);
                givens_matrix[(p+i)+(p+i)*num_dim] = c;
                givens_matrix[(p+i+1)+(p+i+1)*num_dim] = c;
                givens_matrix[(p+i)+(p+i+1)*num_dim] = s;
                givens_matrix[(p+i+1)+(p+i)*num_dim] = -s;

                // matrixB *= givensMatrix;
                // matrixV *= givensMatrix;
                float * matrix_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
                srl_matrix_multiply(matrix_b, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
                srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);

                srl_matrix_multiply(matrix_v, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
                srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_v);

                free(matrix_temp);
                free(givens_matrix);
            }
        }

        if (!flag_zero_diag)
        {
            // GolubKahanSVDStep(matB_2, matP, matQ);
            float * mat_p = (float *)malloc((dimB_2*dimB_2)*sizeof(float));
            float * mat_q = (float *)malloc((dimB_2*dimB_2)*sizeof(float));
            golub_kahan_svd_step(matB_2, mat_p, mat_q, dimB_2);

            for (int i = 0; i < dimB_2; ++i)
                for (int j = 0; j < dimB_2; ++j)
                    matrix_b[(p+i)+(p+j)*num_dim] = matB_2[i+j*dimB_2];

            float * mat_p_full = (float *)malloc((num_dim*num_dim)*sizeof(float));
            float * mat_q_full = (float *)malloc((num_dim*num_dim)*sizeof(float));
            srl_matrix_identity(mat_p_full, num_dim, num_dim);
            srl_matrix_identity(mat_q_full, num_dim, num_dim);

            for (int m = 0; m < dimB_2; ++m)
            {
                for (int n = 0; n < dimB_2; ++n)
                {
                    mat_p_full[(p+m)+(p+n)*num_dim] = mat_p[m+n*dimB_2];
                    mat_q_full[(p+m)+(p+n)*num_dim] = mat_q[m+n*dimB_2];
                }
            }

            float * mat_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
            srl_matrix_multiply(matrix_u, mat_p_full, num_dim, num_dim, num_dim, mat_temp);
            srl_matrix_copy(mat_temp, num_dim*num_dim, matrix_u);

            srl_matrix_multiply(matrix_v, mat_q_full, num_dim, num_dim, num_dim, mat_temp);
            srl_matrix_copy(mat_temp, num_dim*num_dim, matrix_v);

            free(mat_temp);
            free(mat_p_full);
            free(mat_q_full);
            free(mat_p);
            free(mat_q);
        }

        free(matB_2);
    }
}

__device__ void golub_kahan_svd_step(float * matrix_b, float * matrix_p, float * matrix_q, int num_dim)
{
    srl_matrix_identity(matrix_p, num_dim, num_dim);
    srl_matrix_identity(matrix_q, num_dim, num_dim);

    float * matrix_b_t = (float *)malloc((num_dim*num_dim)*sizeof(float));
    srl_matrix_transpose(matrix_b, num_dim, num_dim, matrix_b_t);
    float * mat_t = (float *)malloc((num_dim*num_dim)*sizeof(float));
    srl_matrix_multiply(matrix_b_t, matrix_b, num_dim, num_dim, num_dim, mat_t);

    float mat_c[4];
    mat_c[0] = mat_t[(num_dim-2)+(num_dim-2)*num_dim];
    mat_c[1] = mat_t[(num_dim-2+1)+(num_dim-2)*num_dim];
    mat_c[2] = mat_t[(num_dim-2)+(num_dim-2+1)*num_dim];
    mat_c[3] = mat_t[(num_dim-2+1)+(num_dim-2+1)*num_dim];
    float coeff_b = -(mat_c[0] + mat_c[3]);
    float coeff_c = mat_c[0]*mat_c[3] - mat_c[2]*mat_c[1];
    float eig_value_mat_c_1 = (-coeff_b + srl_sqrt(powf(coeff_b, 2)-4.0*coeff_c)) / 2.0;
    float eig_value_mat_c_2 = (-coeff_b - srl_sqrt(powf(coeff_b, 2)-4.0*coeff_c)) / 2.0;
    float mu = fabsf(eig_value_mat_c_1-mat_c[3]) < fabsf(eig_value_mat_c_2-mat_c[3]) ? eig_value_mat_c_1 : eig_value_mat_c_2;

    float alpha = mat_t[0] - mu;
    float beta = mat_t[num_dim];
    for (std::size_t k = 0; k < num_dim-1; ++k)
    {
        float c, s;
        if (fabsf(beta) < SRL_EPS)
        {
            c = srl_sign(alpha);
            s = 0.0;
        }
        else
        {
            s = -beta / srl_sqrt(powf(alpha, 2)+powf(beta, 2));
            c = alpha / srl_sqrt(powf(alpha, 2)+powf(beta, 2));
        }

        float * givens_matrix = (float *)malloc((num_dim*num_dim)*sizeof(float));
        srl_matrix_identity(givens_matrix, num_dim, num_dim);
        givens_matrix[k+k*num_dim] = c;
        givens_matrix[(k+1)+(k+1)*num_dim] = c;
        givens_matrix[k+(k+1)*num_dim] = s;
        givens_matrix[(k+1)+k*num_dim] = -s;

        // matrixB *= givensMatrix;
        // matrixQ *= givensMatrix;
        float * matrix_temp = (float *)malloc((num_dim*num_dim)*sizeof(float));
        srl_matrix_multiply(matrix_b, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
        srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);

        srl_matrix_multiply(matrix_q, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
        srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_q);

        alpha = matrix_b[k+k*num_dim];
        beta = matrix_b[(k+1)+k*num_dim];
        if (fabsf(beta) < SRL_EPS)
        {
            c = srl_sign(alpha);
            s = 0.0;
        }
        else
        {
            s = -beta / srl_sqrt(powf(alpha,2)+powf(beta,2));
            c = alpha / srl_sqrt(powf(alpha,2)+powf(beta,2));
        }

        givens_matrix[k+k*num_dim] = c;
        givens_matrix[(k+1)+(k+1)*num_dim] = c;
        givens_matrix[k+(k+1)*num_dim] = s;
        givens_matrix[(k+1)+k*num_dim] = -s;

        float * givens_matrix_t = (float *)malloc((num_dim*num_dim)*sizeof(float));
        srl_matrix_transpose(givens_matrix, num_dim, num_dim, givens_matrix_t);
        srl_matrix_multiply(givens_matrix_t, matrix_b, num_dim, num_dim, num_dim, matrix_temp);
        srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_b);

        srl_matrix_multiply(matrix_p, givens_matrix, num_dim, num_dim, num_dim, matrix_temp);
        srl_matrix_copy(matrix_temp, num_dim*num_dim, matrix_p);

        free(givens_matrix);
        free(givens_matrix_t);
        free(matrix_temp);

        if (k < num_dim-2)
        {
            alpha = matrix_b[k+(k+1)*num_dim];
            beta = matrix_b[k+(k+2)*num_dim];
        }
    }

    free(matrix_b_t);
    free(mat_t);
}

__device__ void srl_matrix_identity (float * mat_a, int num_row, int num_column)
{
    for (int i = 0; i < num_row; ++i)
    {
        for (int j = 0; j < num_column; ++j)
        {
            if (i == j)
                mat_a[i+j*num_row] = 1.0;
            else
                mat_a[i+j*num_row] = 0.0;
        }
    }
}

__device__ void srl_matrix_zeros (float * mat_a, int num_row, int num_column)
{
    for (int i = 0; i < (num_row*num_column); ++i)
        mat_a[i] = 0.0;
}

__device__ void srl_matrix_ones (float * mat_a, int num_row, int num_column)
{
    for (int i = 0; i < (num_row*num_column); ++i)
        mat_a[i] = 1.0;
}

__device__ void srl_matrix_copy (const float * mat_src, int length , float * mat_dst)
{
    for (int i = 0; i < length; ++i)
        mat_dst[i] = mat_src[i];
}

__device__ void srl_matrix_add (const float * mat_a, const float * mat_b, int num_row, int num_column, float * mat_c)
{
    for (int i = 0; i < num_row; ++i)
        for (int j = 0; j < num_column; ++j)
            mat_c[i+j*num_row] = mat_a[i+j*num_row] + mat_b[i+j*num_row];
}

__device__ void srl_matrix_subtract (const float * mat_a, const float * mat_b, int num_row, int num_column, float * mat_c)
{
    for (int i = 0; i < num_row; ++i)
        for (int j = 0; j < num_column; ++j)
            mat_c[i+j*num_row] = mat_a[i+j*num_row] - mat_b[i+j*num_row];
}

__device__ void srl_matrix_multiply (const float * mat_a, const float * mat_b, int num_row_a, int num_mid, int num_col_b, float * mat_c)
{
    for (int i = 0; i < num_row_a; ++i)
    {
        for (int j = 0; j < num_col_b; ++j)
        {
            mat_c[i+j*num_row_a] = 0.0;
            for (int k = 0; k < num_mid; ++k)
            {
                mat_c[i+j*num_row_a] += mat_a[i+k*num_row_a] * mat_b[k+j*num_mid];
            }
        }
    }
}

__device__ void srl_matrix_multiplied_by_scalar (const float * mat_a, int num_row, int num_column, float scalar, float * mat_b)
{
    for (int i = 0; i < num_row; ++i)
        for (int j = 0; j < num_column; ++j)
            mat_b[i+j*num_row] = mat_a[i+j*num_row] * scalar;
}

__device__ void srl_matrix_divided_by_scalar (const float * mat_a, int num_row, int num_column, float scalar, float * mat_b)
{
    for (int i = 0; i < num_row; ++i)
        for (int j = 0; j < num_column; ++j)
            mat_b[i+j*num_row] = mat_a[i+j*num_row] / scalar;
}

__device__ void srl_matrix_transpose (const float * mat_a, int num_row, int num_column, float * mat_t)
{
    for (int i = 0; i < num_column; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            mat_t[i+j*num_column] = mat_a[j+i*num_row];
        }
    }
}

__device__ float srl_dot_product (const float * mat_a, const float * mat_b, int length)
{
    float result = 0.0;
    for (int i = 0; i < length; ++i)
        result += mat_a[i]*mat_b[i];

    return (result);
}

__device__ void srl_cross_product (const float * vect_a, const float * vect_b, float * vect_c)
{
    vect_c[0] = vect_a[1]*vect_b[2] - vect_a[2]*vect_b[1];
    vect_c[1] = vect_a[2]*vect_b[0] - vect_a[0]*vect_b[2];
    vect_c[2] = vect_a[0]*vect_b[1] - vect_a[1]*vect_b[0];
}

__device__ void srl_exp_map_se(const float * xi, bool with_rotation_angle, float * mat_transform, float angle)
{

}

__device__ void srl_exp_map_so(const float * omega, bool with_rotation_angle, float * mat_transform, float angle)
{

}

