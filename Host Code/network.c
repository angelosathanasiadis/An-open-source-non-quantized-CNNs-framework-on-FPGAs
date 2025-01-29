#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "parser.h"
#include "data.h"
#include "batchnorm_layer.h"
#include "shortcut_layer.h"

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}

network *load_network(char *cfg, char *weights, int clear)
{
	network *net = parse_network_cfg(cfg);


    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {

    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network *make_network(int n)
{
    network *net = (struct network *) calloc(1, sizeof(network));
    net->n = n;
    net->layers = (struct layer *) calloc((net->n + 1), sizeof(layer));
    net->seen = (size_t *)calloc(1, sizeof(size_t));
    net->t    = (int *)calloc(1, sizeof(int));
    net->cost = (float *)calloc(1, sizeof(float));
//    printf("%d", sizeof(size_t) + sizeof(int) + sizeof(float) + net->n * sizeof(layer));
    return net;
}

//network *make_network(int n)
//{
////	struct network *net =(struct network *)  malloc(sizeof (struct network));
//	struct network *net = (struct network *) calloc(1, sizeof(struct network));
//    net->n = n;
//    net->layers = (struct layer *) calloc(net->n, sizeof(layer));
//    net->seen = (size_t *) calloc(1, sizeof(size_t));
//    net->t    = (int *) calloc(1, sizeof(int));
//    net->cost = (float *) calloc(1, sizeof(float));
////    net->workspace = (float *) malloc(sizeof(float));
////    printf("*net->workspace = %f \n", *net->workspace);
////    printf("net->workspace = %p \n", net->workspace);
//
//    printf("size of network is: %lu \n", sizeof(struct network));
//
//    return net;
//}

//void forward_network(network *netp, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf)
//{
//
//	network net = *netp;
//    int i;
//
////    int time_int = 0;
////    clock_t time_2;
////    clock_t time_3;
////    printf("forward_network\n");
//    for(i = 0; i < net.n; ++i){
////    	printf("forward_network begin\n");
//        net.index = i;
//        layer l = net.layers[i];
//        if(l.delta)
//        {
//            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
//        }
////        printf("forward_network step 1\n");
////        time_2=clock();
////        printf("%d: ",++time_int);
//
//        if(l.type == CONVOLUTIONAL)
//        {
//        	l.forward_conv(l, net, a_buf, b_buf, c_buf);
//        }
//        else
//        {
//        	l.forward(l, net);
//        }
//
//
////        printf("\n");
////        time_3=clock();
////        printf("forward_network step 2\n");
//
//        net.input = l.output;
//
//        if(l.truth) {
//
//            net.truth = l.output;
//        }
//
////        printf("forward_network end\n");
////		printf("%d forward in %f seconds.\n",time_int++, sec(time_3-time_2));
//    }
////    printf("forward_network step 2\n");
////    calc_network_cost(netp);
//}

void update_network(network *netp)
{

    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, a);
        }
    }
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    printf("net.n is %d\n",net.n);
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
        	printf("i is %d\n",i);
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    printf("end\n");
    *net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{

    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}

//float train_network_datum(network *net)
//{
//    *net->seen += net->batch;
//    net->train = 1;
//    forward_network(net);
//    backward_network(net);
//    float error = *net->cost;
//    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
//    return error;
//}

//float train_network_sgd(network *net, data d, int n)
//{
//    int batch = net->batch;
//
//    int i;
//    float sum = 0;
//    for(i = 0; i < n; ++i){
//        get_random_batch(d, batch, net->input, net->truth);
//        float err = train_network_datum(net);
//        sum += err;
//    }
//    return (float)sum/(n*batch);
//}
//
//float train_network(network *net, data d)
//{
//    assert(d.X.rows % net->batch == 0);
//    int batch = net->batch;
//    int n = d.X.rows / batch;
//
//    int i;
//    float sum = 0;
//    for(i = 0; i < n; ++i){
//        get_next_batch(d, batch, i*batch, net->input, net->truth);
//        float err = train_network_datum(net);
//        sum += err;
//    }
//    return (float)sum/(n*batch);
//}

void set_temp_network(network *net, float t)
{
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
    }
}


void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;

    }
}


layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer l ;
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];

    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}


//float *network_predict(network *net, float *input, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf)
//{
//	network orig = *net;
//
//    net->input = input;
//    net->truth = 0;
//    net->train = 0;
//    net->delta = 0;
//
//
//    forward_network(net, a_buf, b_buf, c_buf);
//
//    float *out = net->output;
//    *net = orig;
//    return out;
//}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
//        if(l.type == YOLO){
//            s += yolo_num_detections(l, thresh);
//        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
    detection *dets = (struct detection *) calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = (float *) calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = (float *) calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

//float *network_predict_image(network *net, image im)
//{
//    image imr = letterbox_image(im, net->w, net->h);
//    set_batch_network(net, 1);
//    float *p = network_predict(net, imr.data);
//    free_image(imr);
//    return p;
//}

int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}


void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}


layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}


void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);

    free(net);
}



layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}


