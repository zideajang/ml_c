#define NN_IMPLEMENTATION

// #define NN_MALLOC my_malloc
#include "nn.h"
#include "stdio.h"

int main(int argc, char const *argv[])
{   

    Mat m = mat_alloc(10,10);
    mat_print(m);
    
    return 0;
}
