#define NN_IMPLEMENTATION

// 通过定义 NN_MALLOC 来实现对函数复写
// #define NN_MALLOC my_malloc
#include "nn.h"
#include "stdio.h"
#include<stdlib.h>
#include<time.h>

// void my_malloc

int main(int argc, char const *argv[])
{   
    srand(time(0));
    // 数据
    Mat x = mat_alloc(1,2);

    // MLP 多层感知机
    // 定义第一层
    Mat w1 = mat_alloc(2,2);
    Mat b1 = mat_alloc(1,2);

    Mat a1 = mat_alloc(1,2);

    // 定义第二层(输出层)
    Mat w2 = mat_alloc(2,1);
    Mat b2 = mat_alloc(1,1);
    Mat a2 = mat_alloc(1,1);

    mat_rand(w1,0,1);
    mat_rand(b1,0,1);
    mat_rand(w2,0,1);
    mat_rand(b2,0,1);

    MAT_AT(x,0,0) = 0;
    MAT_AT(x,1,0) = 1;

    mat_dot(a1,x,w1);
    mat_sum(a1,b1);
    mat_sig(a1);

    

    // sigmoidf(x*)

    MAT_PRINT(w1);
    MAT_PRINT(b1);
    MAT_PRINT(w2);
    MAT_PRINT(b2);
    
    return 0;
}
