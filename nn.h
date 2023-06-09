
#ifndef NN_H_
#define NN_H_
#include "stdio.h"
#include "assert.h"
#include <stddef.h>

#ifndef NN_MALLOC
#include "stdlib.h"
#define NN_MALLOC malloc
#endif //NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif //NN_ASSERT

#include<math.h>

// x86_64 Linux

// float d[] ={
//  0, 0, 0
//  0, 1, 1
//  1, 0, 1
//  1, 1, 1
// }

// float d[] = { 0,0,0,1,1,1}
//  Mat di = {.rows = 4, .cols=2 .stride=3 .es=&d[0]}
//  Mat di = {.rows = 4, .cols=1 .stride=3 .es=&d[3]}
// Mat m {.rows = 2,.cols=3, .es=d};
//  {}

typedef struct 
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;
// {1,2,3,4,5,6}
#define MAT_AT(m,i,j) m.es[(i)*(m).stride + (j)]
float rand_float(void);
float sigmoidf(float x);
// 通过 malloc 初始化一个 Matrix
Mat mat_alloc(size_t rows, size_t cols);
// 获取随机数
float rand_float(void);
// 随机初始化矩阵，给出一定范围取值，在 low 和 high 之间
void mat_rand(Mat m,float low, float high);

Mat mat_row(Mat m,size_t row);
// void mat_sub(Mat m,size_t)
void mat_copy(Mat dst, Mat src);

// 矩阵的点乘 a dot b = dst 那么 a 矩阵(m,k) b 矩阵(k,n) 点乘后得到矩阵 dst (m,n)
void mat_dot(Mat dst, Mat a, Mat b);
// dst = dst + a 矩阵
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_fill(Mat m, float x);
void mat_print(Mat m,const char *name);
#define MAT_PRINT(m) mat_print(m,#m)

#endif //NN_H_
#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand()/(float)RAND_MAX;
}
// 1 / (1 + epx(x))
float sigmoidf(float x)
{
    return 1.f /(1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m; 
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = x;
        }
        
    }
    
}
// flaot m[] = {1,2,3,4,5,6}

Mat mat_row(Mat m,size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m,row,0),
    };
}
void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++){
        for (size_t j = 0; j <dst.cols; j++){
            MAT_AT(dst,i,j) = MAT_AT(src,i,j);
        }
    }
}


void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = sigmoidf(MAT_AT(m,i,j));
        }
        
    }
    
}

// grid/block/thread threadIdx 

void mat_dot(Mat dst, Mat a, Mat b){
    assert(a.cols == b.rows);
    size_t n = a.cols;
    assert(dst.rows == a.rows);
    assert(dst.cols ==  b.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j) = 0;
            for (size_t k = 0; k < n; k++)
            {
                MAT_AT(dst,i,j) += MAT_AT(a,i,k) * MAT_AT(b,k,j);
            }
            
        }
        
    }
    
}

// 添加 bias 
void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j) += MAT_AT(a,i,j);
        }
    }
}


void mat_print(Mat m, const char* name){

    printf("%s = [\n",name);
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("    %f ",MAT_AT(m,i,j));
        }
        printf("\n");
    }
    printf("]\n");
}


void mat_rand(Mat m,float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = rand_float() * (high - low)  + low;
        }
    }
}


#endif //NN_IMPLEMENTATION