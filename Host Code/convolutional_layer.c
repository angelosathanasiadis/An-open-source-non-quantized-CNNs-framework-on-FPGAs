#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "batchnorm_layer.h"
#include <omp.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
#pragma omp parallel for
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
#pragma omp parallel for
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
#pragma omp parallel for
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l)
{

    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer l;
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = (float *) calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = (float *) calloc(c/groups*n*size*size, sizeof(float));

    l.biases = (float *) calloc(n, sizeof(float));
    l.bias_updates = (float *) calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = (float *) calloc(l.batch*l.outputs, sizeof(float));
    l.delta  =  (float *) calloc(l.batch*l.outputs, sizeof(float));

    l.forward_conv = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = (float *) calloc(l.nweights, sizeof(float));
        l.cweights =(char *) calloc(l.nweights, sizeof(char));
        l.scales = (float *) calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = (float *) calloc(l.nweights, sizeof(float));
        l.binary_input = (float *) calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = (float *) calloc(n, sizeof(float));
        l.scale_updates = (float *) calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = (float *) calloc(n, sizeof(float));
        l.variance = (float *) calloc(n, sizeof(float));

        l.mean_delta = (float *) calloc(n, sizeof(float));
        l.variance_delta = (float *) calloc(n, sizeof(float));

        l.rolling_mean = (float *) calloc(n, sizeof(float));
        l.rolling_variance = (float *) calloc(n, sizeof(float));
        l.x = (float *) calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = (float *) calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = (float *) calloc(l.nweights, sizeof(float));
        l.v = (float *) calloc(l.nweights, sizeof(float));
        l.bias_m = (float *) calloc(n, sizeof(float));
        l.scale_m = (float *) calloc(n, sizeof(float));
        l.bias_v = (float *) calloc(n, sizeof(float));
        l.scale_v = (float *) calloc(n, sizeof(float));
    }


    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}


void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
#pragma omp parallel for
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
#pragma omp parallel for
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
#pragma omp parallel for
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

//void forward_convolutional_layer(convolutional_layer l, network net, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf)
//{
//    int i, j;
//
////    int time_int = 0;
////    clock_t time;
////    clock_t time_2;
////    clock_t time_3;
//
////    time=clock();
//
//    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
//
//    if(l.xnor){
//        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
//        swap_binary(&l);
//        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
//        net.input = l.binary_input;
//    }
//
//    int m = l.n/l.groups;
//    int k = l.size*l.size*l.c/l.groups;
//    int n = l.out_w*l.out_h;
//    for(i = 0; i < l.batch; ++i){
//        for(j = 0; j < l.groups; ++j){
//            float *a = l.weights + j*l.nweights/l.groups;
////            float *b = net.workspace;
////            printf("step 2\n");
//            float *b =(float *)malloc((k+1)*n*sizeof(float));
////            printf("step 3\n");
////            printf("*b = %f \n", *b);
////            printf("*net.workspace = %f \n", *net.workspace);
////            printf("b = %p \n", b);
////            printf("net.workspace = %p \n", net.workspace);
////            printf("step 2\n");
//
//            float *c = l.output + (i*l.groups + j)*n*m;
//            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
//
//            if (l.size == 1) {
//                b = im;
//            } else {
////            	printf("step 1\n");
//                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
////                printf("step 2\n");
//            }
////            time_2=clock();
////            printf("step mpravo\n");
//            gemm_cpu_kernel(0,0,m,n,k,1,a,k,b,n,1,c,n, a_buf, b_buf, c_buf);
////            time_3=clock();
////
////            printf("%d total in %f seconds.\n",++time_int, sec(time_3-time));
////            printf("%d gemm in %f seconds.\n",time_int, sec(time_3-time_2));
////            printf("%d rest in %f seconds.",time_int, sec(time_2-time));
//            free(b);
//        }
//    }
//
//
//        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
//
//
//    activate_array(l.output, l.outputs*l.batch, l.activation);
//    if(l.binary || l.xnor) swap_binary(&l);
//}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    printf("backward_convolutional_layer\n");

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);


    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }



    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w,
                        l.size, l.stride, l.pad, b);
            }

            printf("problem_1\n");
            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = (image *) calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

