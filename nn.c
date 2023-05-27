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

    Mat a = mat_alloc(2,3);
    Mat b = mat_alloc(3,2);
    Mat m = mat_alloc(2,2);

    mat_rand(a,0,1);
    mat_rand(b,0,2);
    mat_print(a);
    mat_print(b);
    printf("-----------------------\n");
    mat_dot(m,a,b);
    mat_print(m);
    
    return 0;
}
