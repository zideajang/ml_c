#define NN_IMPLEMENTATION

// 通过定义 NN_MALLOC 来实现对函数复写
// #define NN_MALLOC my_malloc
#include "nn.h"
#include "stdio.h"
#include<stdlib.h>
#include<time.h>

// void my_malloc

// 定义 Xor 多层感知机
typedef struct 
{
    Mat a0;
    Mat w1,b1,a1;
    Mat w2,b2,a2;
} Xor;

float forward_xor(Xor m);

float cost(Xor m, Mat ti, Mat to)
{
    // 输入神经网络的ti
    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);

    size_t row_num = ti.rows;
    float c = 0;
    for (size_t i = 0; i < row_num; i++)
    {
        Mat x = mat_row(ti,i);
        Mat y = mat_row(to,i);
        
        mat_copy(m.a0,x);
        forward_xor(m);

        size_t col_num = to.cols;
        for(size_t j = 0; j < col_num; j++)
        {
            float d = MAT_AT(m.a2,0,j) - MAT_AT(y,0,j);
            c += d*d;
        }
    }

    return c/row_num;
    
}

// 前向传播
float forward_xor(Xor m)
{
    mat_dot(m.a1,m.a0,m.w1);
    mat_sum(m.a1,m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2,m.a1,m.w2);
    mat_sum(m.a2,m.b2);
    mat_sig(m.a2);

}

// Xor x1 x2 y 
float td[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};


int main(int argc, char const *argv[])
{   
    srand(time(0));
    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    MAT_PRINT(ti);
    MAT_PRINT(to);

    
    //定义模型
    Xor model;
 
    model.a0 = mat_alloc(1,2);
    model.w1 = mat_alloc(2,2);
    model.b1 = mat_alloc(1,2);
    model.a1 = mat_alloc(1,2);
    model.w2 = mat_alloc(2,1);
    model.b2 = mat_alloc(1,1);
    model.a2 = mat_alloc(1,1);

    // 随机初始化模型
    mat_rand(model.w1,0,1);
    mat_rand(model.b1,0,1);
    mat_rand(model.w2,0,1);
    mat_rand(model.b2,0,1);

    // printf("cost = %f\n",cost(m,ti,to));

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {

            MAT_AT(model.a0,0,0) = i;
            MAT_AT(model.a0,0,1) = j;
            forward_xor(model);
            float y = *model.a2.es;
            printf("%zu ^ %zu = %f \n",i,j,y);
        }
        
    }
    


    return 0;
}
