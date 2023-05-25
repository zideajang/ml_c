
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
    float *es;
} Mat;

#define MAT_AT(m,i,j) m.es[(i)*(m).cols + (j)]

Mat mat_alloc(size_t rows, size_t cols);


float rand_float(void);
void mat_rand(Mat m,float low, float high);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_print(Mat m);


#endif //NN_H_
#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand()/(float)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m; 
}
void mat_dot(Mat dst, Mat a, Mat b){
    (void) dst;
    (void) a;
    (void) b;
}

void mat_sum(Mat dst, Mat a)
{
    (void) dst;
    (void) a;
}


void mat_print(Mat m){
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ",MAT_AT(m,i,j));
        }
        printf("\n");
        
    }

    
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