#include "gemm.h"
#include "utils.h"
#include "wide_vadd.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "darknet.h"
#include <omp.h>

#ifndef SW_EMU
extern int kernel_flag;
#endif

//extern int A_sizes[23];
//extern int B_sizes[23];
//extern int C_sizes[23];
//extern int counter_sz;

//void gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc);

void gemm_bin(int M, int N, int K, float ALPHA,
        char  *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;

    printf("gemm_bin\n");

    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
	printf("random_matrix\n");
    int i;
    float *m = (float *) calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    printf("time_random_matrix\n");
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

//void gemm_nn(int M, int N, int K, float ALPHA,
//        float *A, int lda,
//        float *B, int ldb,
//        float *C, int ldc)
//{
//    int i,j,k;
//    //printf("gemm_nn\n");
////    #pragma omp parallel for
//    for(i = 0; i < M; ++i){
//        for(k = 0; k < K; ++k){
//            register float A_PART = ALPHA*A[i*lda+k];
//            for(j = 0; j < N; ++j){
//                C[i*ldc+j] += A_PART*B[k*ldb+j];
//            }
//        }
//    }
//}
void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,k;
    int j;
    float A_PART;
//    printf("new kernels for M=%d K=%d N=%d lda=%d ldb=%d ldc=%d \n", M, K, N, lda, ldb, ldc);
//    A_sizes[counter_sz] = (M*lda)+K;
//    B_sizes[counter_sz] = (K*ldb)+N;
//    C_sizes[counter_sz] = (M*ldc)+N;
//    printf("A size is: %d\t", A_sizes[counter_sz]);
//    printf("B size is: %d\t", B_sizes[counter_sz]);
//    printf("C size is: %d\n", C_sizes[counter_sz]);
//    counter_sz++;

//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            A_PART = ALPHA*A[i*lda+k];

			for(j = 0; j < N; ++j)
			{
				C[i*ldc+j] += A_PART*B[k*ldb+j];
			}
        }

    }
//    if(kernel_flag)
//    {
//    	kernel_flag=0;
//    }
}


void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    printf("gemm_nt\n");
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    printf("gemm_tn\n");
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    printf("gemm_tt\n");
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("gemm_cpu\n");
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
    	gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
    {
    	printf("deconv\n");
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    }
    else if(!TA && TB)
    {
    	printf("problem\n");
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    }
    else
    {
    	printf("problem\n");
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    }
}



