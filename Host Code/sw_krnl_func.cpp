
#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include <dirent.h>
#include <string.h>
#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdexcept>
#include "batchnorm_layer.h"
#include "dilated_convolutional_layer.h"


extern int kernel_flag;

extern int A_sizes[ARRAYS_SZS_SZ];
extern int B_sizes[ARRAYS_SZS_SZ];
extern int C_sizes[ARRAYS_SZS_SZ];
extern int counter_sz;

extern int krnl_M;
extern int krnl_K;
extern int krnl_N;
extern int krnl_lda;
extern int krnl_ldb;
extern int krnl_ldc;
extern float krnl_var;


void predict_unet_segmenter(cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{
    srand(2222222);
    DIR *dir;
    struct dirent *ent;

    system("v4l2-ctl --device /dev/video0 --set-fmt-video=width=512,height=512,pixelformat=MJPG --stream-mmap --stream-to=../src/unet/test/test.png --stream-count=1");

//    char *cfg = "../src/unet.cfg";
    char *cfg = "../src/unet_custom.cfg";//"/mnt/data/angelos/vitis2019_workspace/unet_darknet_small/src/unet.cfg";//"../../src/unet.cfg";
    char *weights = "../src/unet.backup";//"/mnt/data/angelos/vitis2019_workspace/unet_darknet_small/src/unet.backup"; // You have load your model here

    char dirname[256],resdirname[256], filename[256], resfilename[256];
    strcpy(dirname,"../src/unet/test/");
    //strcpy(dirname,"/mnt/data/angelos/vitis2019_workspace/unet_darknet_small/src/unet/test/");
    strcpy(resdirname,"../src/unet/result/");
    //strcpy(resdirname,"/mnt/data/angelos/vitis2019_workspace/unet_darknet_small/src/unet/result/");
    network *net = load_network(cfg, weights, 0);

    set_batch_network(net, 1);


    if ((dir = opendir (dirname)) != NULL) {
//        clock_t time;
//        clock_t time_2;
//        clock_t time_3;
        char buff[256];
        char *input = buff;
   	 EventTimer et3;


        while ((ent = readdir (dir)) != NULL) {
            if (strstr(ent->d_name, "png")!= NULL) {
                strcpy(filename, dirname);
                strcat(filename, ent->d_name);
                strcpy(resfilename, resdirname);
                strcat(resfilename, ent->d_name);
                printf ("%s\n", filename);
                strncpy(input, filename, 256);
                image im = load_image_color(input, 0, 0);
                float *X = (float *) im.data;

//                time=clock();
                printf ("Start Network Prediction h \n");

                et3.add("prediction");
                float *predictions = network_predict(net, X, a_buf, b_buf, c_buf, d_buf);
                et3.finish();

                printf("End Network Prediction\n");

                et3.print();

                image pred = get_network_image(net);
                image prmask = mask_to_rgb(pred);
                save_image_png(prmask, resfilename);
                show_image(prmask, "orig", 0);

				if(*predictions != 0)
                {
                	printf("Predicted: %f\n", *predictions);
                }
                else
                {
                	printf("No road surface detected\n");
                }


		free_image(prmask);
        }
        }
        closedir (dir);
    } else{
    	printf ("Step could not open directory \n");
        /* could not open directory */
        perror ("");
    }
}

void forward_network(network *netp, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{

	network net = *netp;
    int i;
    EventTimer et;



    for(i = 0; i < net.n; ++i){



        net.index = i;
        layer l = net.layers[i];
        if(l.delta)
        {
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }

//        if(l.type == CONVOLUTIONAL)
//        {
//        	printf("CONVOLUTIONAL layer \n");
//        }
//        else if(l.type == MAXPOOL)
//        {
//        	printf("MAXPOOL layer \n");
//        }
//        else if(l.type == ROUTE)
//        {
//        	printf("ROUTE layer \n");
//        }
//        else
//        {
//        	printf("other layer \n");
//        }
#if(DEBUG_TEST)
        char printf_buf[50] = {0};

        if(l.type == CONVOLUTIONAL)
        {
        	sprintf(printf_buf,"%d convolutional layer",i);
        }
        else if(l.type == DECONVOLUTIONAL)
        {
        	sprintf(printf_buf,"%d deconvolutional layer",i);
        }
        else if(l.type == MAXPOOL)
        {
        	sprintf(printf_buf,"%d maxpool layer",i);
        }
        else if(l.type == SHORTCUT)
        {
        	sprintf(printf_buf,"%d shortcut layer",i);
        }
        else if(l.type == DILATED_CONVOLUTIONAL)
        {
        	sprintf(printf_buf,"%d Dilated Convolutional layer",i);
        }
        else
        {
        	sprintf(printf_buf,"%d other layer",i);
        }

        et.add(printf_buf);
#endif
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL || l.type == DILATED_CONVOLUTIONAL)
        {
        	l.forward_conv(l, net, a_buf, b_buf, c_buf, d_buf);
        }
        else
        {
        	l.forward(l, net);
        }
#if(DEBUG_TEST)
        et.finish();
#endif
//        printf("\n");
//        time_3=clock();
//        printf("forward_network step 2\n");

        net.input = l.output;

        if(l.truth) {

            net.truth = l.output;
        }
    }

#if(DEBUG_TEST)
    std::cout << "--------------- layers times ---------------" << std::endl;

    et.print();
#endif

//    printf("forward_network step 2\n");
//    calc_network_cost(netp);
}

float *network_predict(network *net, float *input, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{

	network orig = *net;

    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;

    forward_network(net, a_buf, b_buf, c_buf, d_buf);

    float *out = net->output;
    *net = orig;

    return out;
}

void forward_convolutional_layer(convolutional_layer l, network net, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{
    int i, j;
	 EventTimer et;
	 et.add("fill_cpu");
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    et.finish();


    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){

            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
//            float *b =(float *)malloc((k+1)*n*sizeof(float));
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;





       	 et.add("im2col_cpu");
            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            et.finish();

            //float *d = (float *)malloc(m*n*sizeof(float));
            float *d = l.output + (i*l.groups + j)*n*m;
            et.add("gemm_cpu_kernel");
            gemm_cpu_kernel(0,0,m,n,k,1,a,k,b,n,1,c,n, d, a_buf, b_buf, c_buf, d_buf);
            et.finish();

            //free(d);
        }
    }
#if(BATCH_NORM)
    et.add("batch_normalize");
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

        et.finish();
#endif
        et.add("activate_array");
    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
    et.finish();
#if(DEBUG_TEST)
    std::cout << "--------------- forward_convolutional_layer execution times ---------------" << std::endl;

    et.print();
#endif
}


void forward_deconvolutional_layer(const layer l, network net, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

	 EventTimer et;
	 et.add("fill_cpu");
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    et.finish();

    for(i = 0; i < l.batch; ++i){
        float *a = l.weights;
        float *b = net.input + i*l.c*l.h*l.w;
        float *c = net.workspace;
        //float *d = (float *)malloc(m*n*sizeof(float));
                    float *d = net.workspace;

        et.add("gemm_cpu_kernel");
        gemm_cpu_kernel(1,0,m,n,k,1,a,m,b,n,0,c,n, d, a_buf, b_buf, c_buf, d_buf);
        et.finish();
        //free(d);

        et.add("col2im_cpu");
        col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output+i*l.outputs);
        et.finish();
    }
#if(BATCH_NORM)
    et.add("batch_normalize");
    if (l.batch_normalize) {
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    et.finish();
#endif
    et.add("activate_array");
    activate_array(l.output, l.batch*l.n*l.out_w*l.out_h, l.activation);
    et.finish();
#if(DEBUG_TEST)
    std::cout << "--------------- forward_deconvolutional_layer execution times ---------------" << std::endl;

    et.print();
#endif
}


void forward_dilated_conv_layer(dilated_convolutional_layer l, network net, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{
    int i, j;
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){                                                                              // XNor-Net architecture
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);      // binarilize weight
        swap_binary(&l);                                                                     // swap weight & binary_weight
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);                        // binarilize input
        net.input = l.binary_input;
    }

    int m = l.n/l.groups;                                // 每组的kernel个数
    int k = l.size*l.size*l.c/l.groups;                  // 每组kernel中元素的个数
    int n = l.out_w*l.out_h;                             // 输出图像每个channel的像素个数
    for(i = 0; i < l.batch; ++i){
    //大循环，batch是一组图片，循环内每次对一张图片卷积
        for(j = 0; j < l.groups; ++j){
        //小循环，每次使用一组weights对一张图像进行卷积
            float *a = l.weights + j*l.nweights/l.groups;   // 第j组第一个卷积核的开头元素
            float *b = net.workspace;                       // re-formated image data
            float *c = l.output + (i*l.groups + j)*n*m;     // 第i个图像在和第j组kernel卷积时输出元素的存放位置
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;    // input data

            if (l.size == 1) {
                b = im;
            } else {
                im2col_dilated_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b, l.dilate_rate); // re-format the input image
            }
            //float *d = (float *)malloc(m*n*sizeof(float));
            float *d = l.output + (i*l.groups + j)*n*m;
            gemm_cpu_kernel(0,0,m,n,k,1,a,k,b,n,1,c,n, d, a_buf, b_buf, c_buf, d_buf);
            //free(d);
        }
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}


void gemm_nn_kernel(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc,
		float BETA,  float *D,
		cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{
//    int j;
//    printf("new kernels for M=%d K=%d N=%d lda=%d ldb=%d ldc=%d \n", M, K, N, lda, ldb, ldc);
    A_sizes[counter_sz] = (M*lda)+K;
    B_sizes[counter_sz] = (K*ldb)+N;
    C_sizes[counter_sz] = (M*ldc)+N;
//    printf("A size is: %d\t", A_sizes[counter_sz]);
//    printf("B size is: %d\t", B_sizes[counter_sz]);
//    printf("C size is: %d\n", C_sizes[counter_sz]);
    counter_sz++;

    krnl_lda = lda;
    krnl_ldb = ldb;
    krnl_ldc = ldc;
    krnl_var = ALPHA;

	 if((M*K) > BUFSIZE_A)
	 {
		 printf("Error\n, BUFSIZE_A %d", M*K );
		 int m_div = ceil(M/((M*K)/(BUFSIZE_A+1)));
		  krnl_M = ceil(M/m_div);
		  krnl_K = K;
		  krnl_N = N;
		  for (int i=0; i<(m_div-1);i++)
		  {
			  unet_kernel(A+((i*krnl_M)*krnl_K), B, C+((i*krnl_M)*krnl_N), D+((i*krnl_M)*krnl_N), 0, BETA, a_buf, b_buf, c_buf, d_buf);
		  }
		  int krnl_M_2 = krnl_M;
		  krnl_M = M - ((m_div-1)*krnl_M);
		  unet_kernel(A+(((m_div-1)*krnl_M_2)*krnl_K), B, C+(((m_div-1)*krnl_M_2)*krnl_N), D+(((m_div-1)*krnl_M_2)*krnl_N), 0, BETA, a_buf, b_buf, c_buf, d_buf);
	 }
	 else
	 {

	    krnl_M = M;
	    krnl_K = K;
	    krnl_N = N;

	    unet_kernel(A, B, C, D, 0, BETA, a_buf, b_buf, c_buf, d_buf);
	 }

}


void gemm_tn_kernel(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc,
		float BETA,  float *D,
		cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{
//    int j;
//    printf("new kernels for M=%d K=%d N=%d lda=%d ldb=%d ldc=%d \n", M, K, N, lda, ldb, ldc);
    A_sizes[counter_sz] = (K*lda)+M;
    B_sizes[counter_sz] = (K*ldb)+N;
    C_sizes[counter_sz] = (M*ldc)+N;
//    printf("A size is: %d\t", A_sizes[counter_sz]);
//    printf("B size is: %d\t", B_sizes[counter_sz]);
//    printf("C size is: %d\n", C_sizes[counter_sz]);
    counter_sz++;


    krnl_M = M;
    krnl_K = K;
    krnl_N = N;
    krnl_lda = lda;
    krnl_ldb = ldb;
    krnl_ldc = ldc;
    krnl_var = ALPHA;

    unet_kernel(A, B, C, D, 1, BETA, a_buf, b_buf, c_buf, d_buf);

}




void gemm_cpu_kernel(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc, float *D,
		cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
//	 EventTimer et;
//	 et.add("fill_C_array");
//    int i, j;
//    for(i = 0; i < M; ++i){
//        for(j = 0; j < N; ++j){
//            C[i*ldc + j] *= BETA;
//        }
//    }
//    et.finish();



    if(!TA && !TB)
    	gemm_nn_kernel(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, BETA, D, a_buf, b_buf, c_buf, d_buf);
    else if(TA && !TB)
    	gemm_tn_kernel(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, BETA, D, a_buf, b_buf, c_buf, d_buf);
    else if(!TA && TB)
    	printf("problem\n");
//        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
    	printf("problem\n");
//        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);

//    std::cout << "--------------- gemm_cpu_kernel fill C array ---------------" << std::endl;
//
//    et.print();
}


