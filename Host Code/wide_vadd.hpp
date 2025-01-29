
#ifndef WIDE_VADD_H
#define WIDE_VADD_H

#define ARRAYS_SZS_SZ 100

#define BUFSIZE_A (1024  * 1024 * 17)//1024//(1024 * 32)//
#define BUFSIZE_B (1024  * 1024 * 65)//1024//(1024 * 32)//
#define BUFSIZE_C (1024  * 1024 * 32)//1024//(1024 * 32)//
#define BUFSIZE_D (1024  * 1024 * 32)//1024//(1024 * 32)//
#define BUFSIZE (1024 *  8)

void unet_kernel( float *ena, float *duo, float *tria, float *tessera, int gemm_flag, float beta, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf);
void init_kernel();
void test_kernel(float *a, float *b, float *c, float test, uint32_t size);
void test_kernel_big(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc,
		float *D);


#endif
