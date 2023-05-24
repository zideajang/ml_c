#include<stdio.h>
#include<stdlib.h>
#include<math.h>


/**
 * x1 * w11 
 * x2 * w21
 * 
 * 
 * 
 * 
 * 
 * */
// (x|y) & ~(x&y)
typedef struct 
{
    float or_w1;
    float or_w2;
    float or_b;
    
    float nand_w1;
    float nand_w2;
    float nand_b;
    
    float and_w1;
    float and_w2;
    float and_b;
} Xor;

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
} 

// MLP
float forward( Xor m, float x1, float x2)
{
    float a = sigmoidf(m.or_w1*x1 + m.or_w2 * x2 + m.or_b);
    float b = sigmoidf(m.nand_w1*x1 + m.nand_w2 * x2 + m.nand_b);

    return sigmoidf(a* m.and_w1 + b * m.and_w2 + m.and_b);
}

typedef float sample[3];
sample xor_train[] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,0},
};

sample or_train [] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,1},

};

sample and_train [] = {
    {0,0,0},
    {1,0,0},
    {0,1,0},
    {1,1,1},

};

sample *train = xor_train;
// sample *train = or_train;
// sample *train = and_train;
// 样本数量
size_t train_count = 4;

// 对于成本函数，不需要知道网络结构
float cost(Xor m)
{
    float result = 0.0f;
    for(size_t i = 0;i < train_count; ++i)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        // 网络结构由前向传播定义
        float y = forward(m,x1,x2);
        float d = y - train[i][2];
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

// 随机初始化模型的参数
Xor rand_xor(void)
{
    Xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();
    
    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();
    
    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();

    return m;
}

// 打印模型的参数
void print_xor(Xor m)
{
    printf("or_w1 = %f\n",m.or_w1);
    printf("or_w2 = %f\n",m.or_w2);
    printf("or_b = %f\n",m.or_b);
    
    printf("nand_w1 = %f\n",m.nand_w1);
    printf("nand_w2 = %f\n",m.nand_w2);
    printf("nand_b = %f\n",m.nand_b);
    
    printf("and_w1 = %f\n",m.and_w1);
    printf("and_w2 = %f\n",m.and_w2);
    printf("and_b = %f\n",m.and_b);
}

// 计算梯度
Xor finite_diff(Xor m, float eps)
{   
    Xor g;
    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c)/eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c )/eps;
    m.or_w2 = saved;


    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c)/eps;
    saved = m.or_b;


    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c)/eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c)/eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c)/eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c)/eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c)/eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c)/eps;
    m.and_b = saved;

    return g;
}

Xor learn(Xor m, Xor g, float rate)
{
    m.or_w1 -= rate * g.or_w1;
    m.or_w2 -= rate * g.or_w2;
    m.or_b -= rate * g.or_b;
    
    m.nand_w1 -= rate * g.nand_w1;
    m.nand_w2 -= rate * g.nand_w2;
    m.nand_b -= rate * g.nand_b;
    
    m.and_w1 -= rate * g.and_w1;
    m.and_w2 -= rate * g.and_w2;
    m.and_b -= rate * g.and_b;

    return m;
}

int main(int argc, char const *argv[])
{
    srand(0);
    Xor m = rand_xor();
    float eps = 1e-1;
    float rate = 1e-1;
    print_xor(m);

    printf("------------------------\n");
    for (size_t i = 0; i < 100 * 1000; i++)
    {   
        
        Xor g = finite_diff(m,eps);
        m = learn(m,g,rate);
        /* code */
        if(i % 1000 == 0){
            // printf("iteration %zu cost = %f\n",i,cost(m));
        }
    }
    

    // printf("cost = %f",cost(m));
    
    for (size_t x = 0; x < 2; x++)
    {
        for (size_t y = 0; y < 2; y++)
        {
            // printf("%zu ^ %zu = %zu\n",x,y,(x|y)&(~(x&y)));
            printf("%zu ^ %zu = %f\n",x,y,forward(m,x,y));
        }
        
    }
    printf("------------------------\n");
    printf("-------- OR neuron -------------\n");
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu | %zu = %f\n",i,j, sigmoidf(m.or_w1*i + m.or_w2 * j + m.or_b));
        }
        
    }
    
    printf("-------- AND neuron -------------\n");
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu & %zu = %f\n",i,j, sigmoidf(m.and_w1*i + m.and_w2 * j + m.and_b));
        }
        
    }
    
    
    printf("-------- NAND neuron -------------\n");
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("~(%zu & %zu) = %f\n",i,j, sigmoidf(m.nand_w1*i + m.nand_w2 * j + m.nand_b));
        }
        
    }
    
    
    
    return 0;
}
