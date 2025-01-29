#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
#include "darknet.h"

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, int output_padding, ACTIVATION activation, int batch_normalize, int adam);
void resize_deconvolutional_layer(layer *l, int h, int w);
void forward_deconvolutional_layer(const layer l, network net, cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf, cl::Buffer d_buf);
void update_deconvolutional_layer(layer l, update_args a);
void backward_deconvolutional_layer(layer l, network net);

#endif

