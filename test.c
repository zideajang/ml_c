#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<assert.h>

// pytorch/tensorflow/mxnet
#define MAT_AT(m,i,j) (m).data[(i)*(m).cols + (j)]



typedef struct 
{
    size_t rows;
    size_t cols;
    float *data;
} Mat;

// 生成浮点随机数
float rand_float(void)
{
    return (float) rand()/(float)RAND_MAX;
}


void mat_sum(Mat dst, Mat a){
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j) += MAT_AT(a,i,j);
        }
        
    }
    

}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = malloc(sizeof(*m.data)*rows*cols);
    // if(m.data == NULL){
    //     return ;
    // }
    return m; 
}

void mat_print(Mat m)
{
    // 遍历 rows
    for (size_t i = 0; i < m.rows; i++)
    {
        // 遍历 cols
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
            MAT_AT(m,i,j) = rand_float() * (high - low) + low;
        }
    }
}
// 
// shift + alt + arrow

int main(int argc, char const *argv[])
{

    // Mat m = {.rows=2,.cols=3};
    srand(time(0));
    Mat a = mat_alloc(2,3);
    Mat b = mat_alloc(3,2);
    Mat m = mat_alloc(2,2);
    mat_rand(a,1,2);
    mat_rand(b,1,2);

    printf("------------------ before sum -----------\n");
    mat_print(a);
    mat_print(b);

    mat_sum(m,a);
    printf("------------------ after  sum -----------\n");
    mat_print(m);


    return 0;
}

