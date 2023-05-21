#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// 添加激活函数，

float train[][2] = {
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8},
};

#define train_count sizeof(train)/sizeof(train[0])

// x1 x2 x3,...
// w1 w2 w3,...
// y = w1x1 + w2x2 + w3x3 ... + b
// bias offset 先验值，起点

float cost(float w, float b)
{
    float result = 0.0f;
    for(size_t i = 0;i < train_count; ++i)
    {
        float x = train[i][0];
        float y = x * w + b;
        // printf("actual: %f expected: %f\n",y,train[i][1]);
        float d = y - train[i][1];
        result += d*d;

    }
    // (y - y_hat)^2/n

    result /= train_count;
    return result;
}

float rand_float(void)
{
    return (float) rand()/(float)RAND_MAX;
}

int main(int argc, char const *argv[])
{
    srand(time(0));
    // srand(69);
    float w = rand_float()*10.0f;
    float b = rand_float()*5.0f;

    float eps = 1e-3;
    float rate = 1e-3;

    printf("result:%f\n",cost(w,b));
    for (size_t i = 0; i < 220; i++)
    {
        float c = cost(w,b);
        float dw = (cost(w + eps,b) - c)/eps;
        float db = (cost(w,b + eps) - c)/eps;
        w -= rate * dw;
        b -= rate * db;
        printf("cost = %f, w = %f, b= %f\n",cost(w,b),w,b);
        /* code */
    }
    printf("--------------------\n");
    printf("w:%f\n,b:%f",w,b);
    
    return 0;
}
