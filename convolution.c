#include "convolution.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "logging.h"

//add padding to blob
BLOB* pad(BLOB* in, int pad){

    //create output blob
    BLOB* out = blob_calloc(in->d, in->h+2*pad, in->w+pad*2);

    //copy non-padded input into output blob
    for(int z=0;z<in->d;z++)
       for(int y=0;y<in->h;y++)
          for(int x=0;x<in->w;x++)
              blob_data(out,z,y+pad,x+pad)= blob_data(in,z,y,x);

    //return pointer to padded blob
    return out;
}


BLOB* load_weights(BLOB* b, conv_param_t* p){

    //open weights file for reading
    FILE* fp = fopen(p->weights, "rb");
    if(fp==NULL)
        error("could not open file %s for reading\out_x",p->weights);

    //for fully connected layers the kernel size is equal to the input size
    int Ky=(p->fc)?b->h:p->Ky;
    int Kx=(p->fc)?b->w:p->Kx;

    //allocate 3D blob, and emulate 4D in KxKy later
    BLOB* w=blob_alloc(p->num_out, b->d/p->group, Ky*Kx);

    //fill 4D weight structure
    for(int group_id=0;group_id<p->group;group_id++)
        for(int out_depth=group_id*(p->num_out/p->group);out_depth<(group_id+1)*(p->num_out/p->group);out_depth++)
            for(int i=group_id*(b->d/p->group);i<(group_id+1)*(b->d/p->group);i++)
                //note: each output map has only  b->d/p->group input maps. Hence the absolute index of i is subtracted when storing in w!
                if((int)fread( &(blob_data(w,out_depth,i-group_id*(b->d/p->group),0)),sizeof(float),Ky*Kx, fp)!=Ky*Kx)
                    error("loading weights from file %s\out_x", p->weights);

    //close file
    fclose(fp);

    //return weight blob
    return w;
}

float* load_1d(const char* fname, size_t num){

    //open file for reading
    FILE* fp = fopen(fname, "rb");
    if(fp==NULL)
        error("could not open file %s for reading\out_x",fname);

    //read in array
    float* arr= (float*) malloc(sizeof(float)*num);
    if(fread(arr,sizeof(float), num, fp)!=num)
        error("loading data from file %s\out_x", fname);

    //close file
    fclose(fp);

    return arr;
}

//convolution, NOTE: destructive of BLOB* in. duplicate if further required!
BLOB* convolution(BLOB* input, conv_param_t* p){

    //use local pointer
    BLOB* in = input;

    //padding of input if required
    if(p->pad!=0)
        in = pad(in, p->pad);

    //if fully connected, the kernel size is set to the image size
    int Ky=(p->fc)?in->h:p->Ky;
    int Kx=(p->fc)?in->w:p->Kx;

    //create blob to hold output
    int height=(int)floor(((float)in->h - (float)Ky)/(float)p->Sy)+1;
    int width =(int)floor(((float)in->w - (float)Kx)/(float)p->Sx)+1;
    BLOB* out;

    //load bias if required
    if(p->bias==NULL){
        //zero init
        out = blob_calloc(p->num_out, height, width);
    }else{
        //not required to calloc
        out = blob_alloc(p->num_out, height, width);

        //load bias values from file
        float* bias =load_1d(p->bias, p->num_out);

        //set bias or init with zeroes
        for(int out_depth=0;out_depth<out->d;out_depth++)
            for(int out_y=0;out_y<out->h;out_y++)
                for(int out_x=0;out_x<out->w;out_x++)
                    blob_data(out,out_depth,out_y,out_x)=bias[out_depth];

        //cleanup bias
        free(bias);
    }

    //load weights
    BLOB* w = load_weights(in, p);
    
    // printf("out w : %i  h : %i d : %i  \n",out->w,out->h,out->d);
    // printf("in w : %i  h : %i d : %i  \n",in->w,in->h,in->d);
    // printf("Ky : %i Kx : %i  \n ",Ky,Kx);

    
    //perform convolution
    if(out->w == 1)
    {
        if(in->w==1)
        {
            for(int i=0;i<500;i++)
                for(int out_depth=0;out_depth<10;out_depth++)
                    out->data[out_depth] +=
                        in->data[i] * 
                        blob_data(w, out_depth, i, 0);

        }
        else
        {
            for(int i=0;i<50;i++)
                for(int out_depth=0;out_depth<500;out_depth++)
                    {
                    for(int kx=0;kx<4;kx++)
                        //note: absolute starting i is subtracted for the weights, see load_weights function for more info
                        out->data[out_depth] +=
                            blob_data(in, i, 0, kx) * 
                            blob_data(w, out_depth, i, kx);

                    
                    for(int kx=0;kx<4;kx++)
                        //note: absolute starting i is subtracted for the weights, see load_weights function for more info
                        out->data[out_depth] +=
                            blob_data(in, i, 1, kx) * 
                            blob_data(w, out_depth, i, 4 + kx);
                    for(int kx=0;kx<4;kx++)
                        //note: absolute starting i is subtracted for the weights, see load_weights function for more info
                        out->data[out_depth] +=
                            blob_data(in, i, 2, kx) * 
                            blob_data(w, out_depth, i, 8 + kx);
                    for(int kx=0;kx<4;kx++)
                        //note: absolute starting i is subtracted for the weights, see load_weights function for more info
                        out->data[out_depth] +=
                            blob_data(in, i, 3, kx) * 
                            blob_data(w, out_depth, i, 12 + kx);
                    
                    }
        }
    }
    else
    {
        for(int i=0;i<in->d;i++)
            for(int out_depth=0;out_depth<out->d;out_depth++)
                for(int out_y=0;out_y<out->h;out_y++)
                    for(int out_x=0;out_x<out->w;out_x++)
                        for(int ky=0;ky<5;ky++)
                            for(int kx=0;kx<5;kx++)
                                //note: absolute starting i is subtracted for the weights, see load_weights function for more info
                                blob_data(out,out_depth,out_y,out_x)+=
                                    blob_data(in, i, out_y+ky, out_x+kx) * 
                                    blob_data(w, out_depth, i, ky*Kx + kx);
    }


    //free weights
    blob_free(w);

    //done with padded blob, free
    if(p->pad!=0)
        blob_free(in);

    //perform batchnorm if needed
    if(p->bn_mean!=NULL){


        //load batchnorm mean and variance
        float* mean = load_1d(p->bn_mean, out->d);
        float* var  = load_1d(p->bn_var, out->d);

        //batchnorm
        for(int out_depth=0;out_depth<out->d;out_depth++)
            for(int out_y=0;out_y<out->h;out_y++)
                for(int out_x=0;out_x<out->w;out_x++)
                    blob_data(out,out_depth,out_y,out_x)= (blob_data(out,out_depth,out_y,out_x) - mean[out_depth])/sqrtf(var[out_depth]+p->bn_eps);

        //free mean and variance
        free(mean);
        free(var);
    }

    //perform scale if needed
    if(p->scale!=NULL){
        //load scale parameters
        float* scale = load_1d(p->scale, out->d);
        float* scale_bias = load_1d(p->scale_bias, out->d);

        //scale
        for(int out_depth=0;out_depth<out->d;out_depth++)
            for(int out_y=0;out_y<out->h;out_y++)
                for(int out_x=0;out_x<out->w;out_x++)
                    blob_data(out,out_depth,out_y,out_x) = blob_data(out,out_depth,out_y,out_x)*scale[out_depth] + scale_bias[out_depth];

        //free parameters
        free(scale);
        free(scale_bias);
    }

    //perform relu
    if(p->relu==true)
        for(int i=0;i<blob_size(out); i++)
            out->data[i] =  fmax(0.0f, out->data[i]);

    //return output
    return out;
}
