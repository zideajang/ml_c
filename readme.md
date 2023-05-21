# C 语言实现机器学习
## 不能再简单一个小实例
- 模拟一个函数
- 输入数据输出参数

```c
#include<stdio.h>

int main(int argc, char const *argv[])
{
    printf("hello machine learning");
    return 0;
}
```

```c
float train[][2] = {
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8},
};

```
### 任何建模
- 抽象

### 任务
现在有一堆数据 (x,y) 建立模型 y = wx 我们通过数据学习到 w 参数

GPT4 的参数量 1 000 000 000 000 参数
我们 w 1 一个参数
// y = w*x 


```c
float rand_float(void)
{
    return (float) rand()/(float)RAND_MAX;
}
```

```c
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// 求导

float train[][2] = {
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8},
};

#define train_count sizeof(train)/sizeof(train[0])

float cost(float w)
{
    float result = 0.0f;
    for(size_t i = 0;i < train_count; ++i)
    {
        float x = train[i][0];
        float y = x * w;
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
    // srand(time(0));
    srand(69);
    float w = rand_float()*10.0f;

    float eps = 1e-3;
    float rate = 1e-3;


    printf("result:%f\n",cost(w));
    for (size_t i = 0; i < 300; i++)
    {
        float dcost = (cost(w + eps) - cost(w))/eps;
        w -= rate * dcost;
        printf("cost = %f, w = %f\n",cost(w),w);
        /* code */
    }
    
    printf("w = %f\n",w);
    printf("hello machine learning");
    return 0;
}
```


创建两个文件 twice.c 和 gates.c

## 激活函数

```c
float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
} 
```
- sigmoid 函数后面添加 f 表示这个函数精度为单精度 float 如果不加 f 这表示 double 这个 c 语言其他库保持一致
- 
测试 sigmoid 函
```c
for(float x =-10.f; x <= 10.f; x += 1.0f){
    printf("%f => %f\n",x ,sigmoidf(x));
}

```