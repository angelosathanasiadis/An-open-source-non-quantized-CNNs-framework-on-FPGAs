

#include "event_timer.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <sys/time.h>
#include <omp.h>
#include <math.h>

// Xilinx OpenCL and XRT includes
#include "xilinx_ocl.hpp"

#include "wide_vadd.hpp"

//#ifdef HW_EMU
//#define BUFSIZE (1024 * 4)//1024//(1024 * 32)//

#define ACCESS_K 64
#define ACCESS_N 36

#define UNROLL_N 2
#define UNROLL_K 4


extern swm::XilinxOcl xocl;
extern cl::CommandQueue q ;
extern cl::Kernel krnl;

extern int krnl_M;
extern int krnl_K;
extern int krnl_N;
extern int krnl_lda;
extern int krnl_ldb;
extern int krnl_ldc;
extern float krnl_var;

void transposeMatrix(int rows, int cols, float *A) {
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
		if (i<j)
		{
		    float temp = A[i*cols+j];
		    A[i*cols+j] = A[j*rows+i];
		    A[j*rows+i] = temp;
		}
        }
    }
}

void copyMatrix(int rows, int cols, float *source, float *dest) {
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
        	dest[i*cols+j] = source[i*cols+j];
        }
    }
}


void vadd_sw(float *a, float *b, float *c, uint32_t size, float test)
{
    for (uint32_t i = 0; i < size; i++) {
    	//c[i] = a[i] + b[i];
    	c[i] = a[i] + test*b[i];
    }
}

void test_kernel(float *a, float *b, float *c, float test, uint32_t size)
{
	 EventTimer et;
	 bool verified = true;
	 float temp;

	 et.add("Verify the results");
    for (uint32_t i = 0; i < size; i++)
    {

    	temp = a[i] + test*b[i];

    	if (temp != c[i])
    	{
    		verified = false;
    		std::cout << "ERROR: software and hardware vadd do not match: "
    				<< temp << "!=" << c[i] << " at position " << i << std::endl;
    		break;
    	}
    }
    et.finish();

	 std::cout << "--------------- Key kernel test times ---------------" << std::endl;
	 et.print();

	 if (verified) {
	     std::cout
	         << std::endl
	         << "OCL-mapped contiguous buffer example complete!"
	         << std::endl
	         << std::endl;
	 }
	 else {
	     std::cout
	         << std::endl
	         << "OCL-mapped contiguous buffer example complete! (with errors)"
	         << std::endl
	         << std::endl;
	 }
}

void test_kernel_big(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc,
		float *D)
{
	 EventTimer et;
	 bool verified = true;

	 et.add("Verify the results");
    for(int m = 0; m < M; ++m){
        for(int k = 0; k < K; ++k){
            float A_PART = ALPHA*A[m*lda+k];

			for(int j = 0; j < N; ++j)
			{

				C[m*ldc+j] += A_PART*B[k*ldb+j];
			}
        }
    }

    for (int i=0;i<BUFSIZE_D;i++)
    {
    	if (D[i] != C[i])
    	{
    		verified = false;
    		std::cout << "ERROR: software and hardware vadd do not match: "
    				<< D[i] << "!=" << C[i] << " at position " << i << std::endl;
    		break;
    	}
    }
    et.finish();


	std::cout << "--------------- Key kernel test times ---------------" << std::endl;
	et.print();

	if (verified) {
	    std::cout
	        << std::endl
	        << "OCL-mapped contiguous buffer example complete!"
	        << std::endl
	        << std::endl;
	}
	else {
	    std::cout
	        << std::endl
	        << "OCL-mapped contiguous buffer example complete! (with errors)"
	        << std::endl
	        << std::endl;
	}
}




void unet_kernel( float *ena, float *duo, float *tria, float *tessera, int gemm_flag, float beta, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)//cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf,
{
// Initialize an event timer we'll use for monitoring the application

	if(gemm_flag)
	{

		transposeMatrix(krnl_M, krnl_K, ena);
	}



	copyMatrix(krnl_M, krnl_K, tria, tessera);

	int start_row=0;

	 EventTimer et;

	 int aags=0;
	 et.add("Set kernel arguments");
	 krnl.setArg(aags++, a_buf);
	 krnl.setArg(aags++, b_buf);
	 krnl.setArg(aags++, c_buf);
	 krnl.setArg(aags++, d_buf);
	 krnl.setArg(aags++, krnl_var);
	 krnl.setArg(aags++, start_row);
	 krnl.setArg(aags++, krnl_M);
	 krnl.setArg(aags++, krnl_K);
	 krnl.setArg(aags++, krnl_N);
	 krnl.setArg(aags++, beta);
	 et.finish();

	 if((krnl_M*krnl_K) > BUFSIZE_A)
	 {
		 printf("Error\n, BUFSIZE_A %d", krnl_M*krnl_K );
	 }
	 if((krnl_K*krnl_N) > BUFSIZE_B)
	 {
		 printf("Error\n, BUFSIZE_B %d", krnl_K*krnl_N );
	 }
	 if((krnl_M*krnl_N) > BUFSIZE_C)
	 {
		 printf("Error\n, BUFSIZE_C %d", krnl_M*krnl_N );
	 }
	 if((krnl_M*krnl_N) > BUFSIZE_C)
	 {
		 printf("Error\n, BUFSIZE_C %d", krnl_M*krnl_N );
	 }

	 et.add("Map buffers to userspace pointers");
	ena = (float *)q.enqueueMapBuffer(a_buf,
												  CL_TRUE,
												  CL_MAP_WRITE,
												  0,
												  krnl_M*krnl_K * sizeof(float));
	duo = (float *)q.enqueueMapBuffer(b_buf,
												  CL_TRUE,
												  CL_MAP_WRITE,
												  0,
												  krnl_K*krnl_N * sizeof(float));
	 tria = (float *)q.enqueueMapBuffer(c_buf,
	                                              CL_TRUE,
												  CL_MAP_WRITE,
	                                              0,
												  krnl_M*krnl_N * sizeof(float));
	 tessera = (float *)q.enqueueMapBuffer(d_buf,
	                                              CL_TRUE,
												  CL_MAP_WRITE,
	                                              0,
												  krnl_M*krnl_N * sizeof(float));
	 et.finish();




 // Send the buffers down to the Alveo card
 et.add("Memory object migration enqueue");
 cl::Event event_sp;
 q.enqueueMigrateMemObjects({a_buf, b_buf, c_buf, d_buf}, 0, NULL, &event_sp);
 clWaitForEvents(1, (const cl_event *)&event_sp);
 et.finish();

/* et.add("Set Read back computation results");

 tessera = (float *)q.enqueueMapBuffer(d_buf,
                                              CL_TRUE,
											  CL_MAP_READ,
                                              0,
											  BUFSIZE_D * sizeof(float));
 et.finish(); */

 et.add("OCL Enqueue task");
 q.enqueueTask(krnl, NULL, &event_sp);
 et.finish();
 et.add("Wait for kernel to complete");
 clWaitForEvents(1, (const cl_event *)&event_sp);
 et.finish();
 // Migrate memory back from device+

 et.add("Set Readback Computation Results");
 q.enqueueMigrateMemObjects({c_buf, d_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
 et.finish();

// et.add("Software VADD run");
// test_kernel_big(krnl_M, krnl_N, krnl_K, krnl_var, ena, krnl_lda, duo, krnl_ldb, tria, krnl_ldc, tessera);
//	et.finish();
	if((int(ceil(krnl_K/ACCESS_K)))%2 == 0)
	{
		copyMatrix(krnl_M, krnl_K, tessera, tria);
	}


 et.add("UNmap buffers from userspace pointers");
 q.enqueueUnmapMemObject(a_buf, ena);
 q.enqueueUnmapMemObject(b_buf, duo);
 q.enqueueUnmapMemObject(c_buf, tria);
 q.enqueueUnmapMemObject(d_buf, tessera);
// free(tessera);
 et.finish();


// std::cout << "--------------- Key execution times ---------------" << std::endl;
// et.print();
 //test_kernel_big(krnl_M, krnl_N, krnl_K, krnl_var, ena, krnl_lda, duo, krnl_ldb, tria, krnl_ldc, tessera);

 }

void init_kernel() {

//	 std::cout << "-- Parallelizing the Data Path --" << std::endl
//	           << std::endl;

	 EventTimer et;

	 printf("-- Parallelizing the Data Path --\n\n");
	 // Initialize the runtime (including a command queue) and load the
	 // FPGA image
	 std::cout << "Loading binary_container_1.xclbin to program the board" << std::endl
	           << std::endl;
	 printf("Loading binary_container_1.xclbin to program the board \n\n");
	 et.add("OpenCL Initialization");

	 // This application will use the first Xilinx device found in the system

	 xocl.initialize("binary_container_1.bin");
     q = xocl.get_command_queue();
	 krnl    = xocl.get_kernel("wide_vadd");
	 et.finish();

	 std::cout << "Running kernel test XRT-allocated buffers and wide data path:" << std::endl
	           << std::endl;

	 et.print();

}



//int main(int argc, char *argv[])
//{
//
//	float ena[2*BUFSIZE];
//	float duo[2*BUFSIZE];
//	float tria[2*BUFSIZE] = {0};
//    for (int i = 0; i < 2*BUFSIZE; i++) {
//    	ena[i] = i;
//        duo[i] = 2 * i;
//    }
//    int test12131 = 50;
//    float test =2;
//
//    init_kernel();
//
//	 std::cout << "1st time begin" << std::endl;
//	unet_kernel( ena, duo, tria, test, BUFSIZE); //a_buf, b_buf, c_buf, d_buf,
//	std::cout << "1st time end" << std::endl;
//
//	std::cout << "2nd time begin" << std::endl;
//	unet_kernel( &ena[BUFSIZE], &duo[BUFSIZE], &tria[BUFSIZE], test, test12131);//a_buf, b_buf, c_buf, d_buf,
//	std::cout << "2nd time end" << std::endl;
//
////	unet_kernel();
//}


