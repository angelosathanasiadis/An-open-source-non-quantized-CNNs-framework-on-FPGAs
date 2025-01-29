#ifndef GEMM_H
#define GEMM_H

#include "darknet.h"

void gemm_bin(int M, int N, int K, float ALPHA,
        char  *A, int lda,
        float *B, int ldb,
        float *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);



#ifndef SW_EMU
//void support(float *a, float *b, float *c, float a_part, int N, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf);
#endif

#endif
