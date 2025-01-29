
#ifndef SW_KRNL_FUNC
#define SW_KRNL_FUNC

#include "darknet.h"


void predict_unet_segmenter(cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf);
void gemm_cpu_kernel(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc, float *D,
		cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf);


#endif /* SW_KRNL_FUNC */
