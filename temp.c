#include<stdio.h>
#include<stdlib.h>

typedef struct 
{
    
} Mat;


int main(int argc, char const *argv[])
{
    float td[] = {
        0,0,0,
        0,1,1,
        1,0,1,
        1,1,0,
    };

    float *fptr;
    fptr = td;
    printf("%f",*(fptr  + 2));
    return 0;
}
