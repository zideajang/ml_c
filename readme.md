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
float train[][2] = {
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8},
};

```

```c
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
```

```c
float rand_float(void)
{
    return (float) rand()/(float)RAND_MAX;
}
```

```

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

### 前向传播
```c
float forward(float w1,float w2, float b){
 for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu | %zu = %f\n",i,j,sigmoidf(i  * w1 + j * w2 + b));
        }
        
    }
    
}
```


### XOR

```c
int main(int argc, char const *argv[])
{
    
    for (size_t x = 0; x < 2; x++)
    {
        for (size_t y = 0; y < 2; y++)
        {
            printf("%zu ^ %zu = %zu\n",x,y,(x|y)&(~(x&y)));
        }
        
    }
    
    return 0;
}

```

访问每一个元素的宏
```c
#define MAT_AT(m,i,j) m.es[(i)*(m).cols + (j)]
```

关于自定义引入

```c
#ifndef NN_MALLOC
#include "stdlib.h"
#define NN_MALLOC malloc
#endif //NN_MALLOC
```
通过在 `nn.h` 头文件将 malloc 文件定义为宏，如果在引入该头文件的 c 文件可以自定义 NN_MALLOC 方法来替换现有 malloc 实现对方法的复写

```c
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

```
通过 malloc 来为一个矩阵分配内存空间来保存数据，数据在内存中是连续存储的，rows 和 cols 只是告诉我们应该以什么样的视角来看数据，也就是如何读取数据

#### 实现打印矩阵方法
```c
void mat_print(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ",m.data[i *m.cols + j]);
        }

        printf("\n");
    }
    
}
```

定义一个宏来替换 `m.data[i *m.cols + j]`
```c
#define MAT_AT(m,i,j) (m).data[(i)*(m).cols + (j)]

```