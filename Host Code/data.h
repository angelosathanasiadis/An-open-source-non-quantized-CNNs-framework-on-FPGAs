#ifndef DATA_H
#define DATA_H
#include <pthread.h>

#include "darknet.h"
#include "list.h"
#include "image.h"

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}

void print_letters(float *pred, int n);
data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
data load_data_super(char **paths, int n, int m, int w, int h, int scale);


data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);

void get_random_batch(data d, int n, float *X, float *y);
data get_data_part(data d, int part, int total);
data get_random_data(data d, int num);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
void fill_truth(char *path, char **labels, int k, float *truth);

#endif
