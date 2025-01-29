//
//#include "darknet.h"
//#include <time.h>
//#include <assert.h>
//#include <dirent.h>
//#include <string.h>
//
//
//void predict_unet_segmenter(cl::Buffer a_buf, cl::Buffer b_buf, cl::Buffer c_buf)
//{
//    srand(2222222);
//    DIR *dir;
//    struct dirent *ent;
//    char *cfg = "/mnt/data/angelos/vitis2019_workspace/unet_darknet_small/src/unet.cfg";//"../../src/unet.cfg";
//    char *weights = "/mnt/data/angelos/vitis2019_workspace/unet_darknet_small/src/unet.backup"; // You have load your model here
//
//    char dirname[256],resdirname[256], filename[256], resfilename[256];
//    strcpy(dirname,"/mnt/data/angelos/vitis2019_workspace/unet_darknet_small/src/unet/test/");
//    strcpy(resdirname,"/mnt/data/angelos/vitis2019_workspace/unet_darknet_small/src/unet/result/");
//    network *net = load_network(cfg, weights, 0);
//
//    set_batch_network(net, 1);
//
//    if ((dir = opendir (dirname)) != NULL) {
////        clock_t time;
////        clock_t time_2;
////        clock_t time_3;
//        char buff[256];
//        char *input = buff;
//
//        while ((ent = readdir (dir)) != NULL) {
//            if (strstr(ent->d_name, "png")!= NULL) {
//                strcpy(filename, dirname);
//                strcat(filename, ent->d_name);
//                strcpy(resfilename, resdirname);
//                strcat(resfilename, ent->d_name);
//                printf ("%s\n", filename);
//                strncpy(input, filename, 256);
//                image im = load_image_color(input, 0, 0);
//                float *X = (float *) im.data;
//
////                time=clock();
//                printf("step 1\n");
//
//                float *predictions = network_predict(net, X, a_buf, b_buf, c_buf);
//
////                time_2=clock();
//                printf("step 2\n");
////                time_2=clock();
//                image pred = get_network_image(net);
//                image prmask = mask_to_rgb(pred);
//                save_image_png(prmask, resfilename);
//                show_image(prmask, "orig", 0);
////                time_3=clock();
////				printf("%s: Predicted in %f seconds.\n", input, sec(time_3-time));
////				printf("%s: network_predict in %f seconds.\n", input, sec(time_2-time));
////				printf("%s: the rest in %f seconds.\n", input, sec(time_3-time_2));
//
//				if(*predictions != 0)
//                {
//                	printf("Predicted: %f\n", *predictions);
//                }
//                else
//                {
//                	printf("No road surface detected\n");
//                }
//
//
//		free_image(prmask);
//        }
//        }
//        closedir (dir);
//    } else{
//        /* could not open directory */
//        perror ("");
//    }
//}
//
//
