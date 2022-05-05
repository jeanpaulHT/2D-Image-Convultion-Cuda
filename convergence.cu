#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgcodecs.hpp>

// #define N 18
// #define DIM N

#define KERNEL_SIZE 3
#define KS KERNEL_SIZE

__global__ void dotProduct(int * img_arr, int* kernel,int* res, int height, int width){
    
    int x =  blockIdx.x * blockDim.x + threadIdx.x;
    int y =  blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= height - KS / 2|| y >= width - KS / 2) return ;
    if(x < KS/2 ||  y < KS/2 ) {res[x * width + y] = 0; return;}


    int val = {};
    int temp_x = 0, temp_y = 0;

    for(int i = x - KS/2; i <= x + KS/2;i++){
        
        for(int j = y - KS/2; j <= y +KS/2; j++){
            val += (img_arr[i * width + j] *  kernel[temp_x * KS + temp_y]);
            // printf("(%d,%d)", img_arr[i * width + j], kernel[temp_x * KS + temp_y]);
            temp_y++;

        }
        // printf("\n");

        temp_x++;
        temp_y = 0;
    }

    res[x * width + y] = val;
}


int main(){
    
    // constexpr int width = 1920, height = 2560;
    cv::Mat img = cv::imread("in/nia.png");

    int height = img.rows; //img.rows  1000
    int width = img.cols; //img.cols    700
     

    int* R = new int[width * height];
    int* G = new int[width * height];
    int* B = new int[width * height];

    // int (*R)[width] = new int[height][width];
    // int (*G)[width] = new int[height][width];
    // int (*B)[width] = new int[height][width];

    printf("%d,%d \n", img.rows, img.cols);


  
    cv::Mat resultImage = cv::Mat::zeros(img.size(), CV_8UC3);
    
    
    for(int i = 0; i < img.rows; i++ ) 
    { 
        for(int j = 0; j < img.cols; j++ ) 
        {
            R[i*img.cols + j] = img.at<cv::Vec3b>(i,j)[2];
            G[i*img.cols + j] = img.at<cv::Vec3b>(i,j)[1];
            B[i*img.cols + j] = img.at<cv::Vec3b>(i,j)[0];
        }
    }

    


    
    // dimension of the image array, assume that is a square
    // for the calculation of threads
    int N  = std::max(width, height);



    // int (*resultR)[width] = new int[height][width];
    // int (*resultG)[width] = new int[height][width];
    // int (*resultB)[width] = new int[height][width];

    int* resultR = new int[width * height];
    int* resultG = new int[width * height];
    int* resultB = new int[width * height];


    int kernel[KS][KS] = {
        {-1,0,1},
        {-1,0,1},
        {-1,0,1}
    };



    
    // Calculating <numBlocks, Threads per block>
    int tpb = min(16, N); //thread per block
    int numBlocks = ceil(float(N) / tpb);

    dim3 blockDim (numBlocks, numBlocks);
    dim3 threadsPerBlock(tpb,tpb);



    // // Allocating space in device memory
    int *d_kernel;
    int *d_imgR, *d_imgG, *d_imgB;
    int *d_resR, *d_resG, *d_resB;
    
    int bytes = sizeof(int);
    int bytes_n = height * width * bytes;
    int kbytes_n = KS * KS * bytes;
    



    cudaMalloc( (void**)  &d_imgR, bytes_n ) ;
    cudaMalloc( (void**)  &d_resR, bytes_n ) ;

    cudaMalloc( (void**)  &d_imgG, bytes_n );
    cudaMalloc( (void**)  &d_resG, bytes_n );



    cudaMalloc( (void**)  &d_imgB, bytes_n );
    cudaMalloc( (void**)  &d_resB, bytes_n );


    cudaMalloc( (void**)  &d_kernel, kbytes_n );


    //Copy CPU data to Device memory
    cudaMemcpy(d_imgR, R, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgG, G, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgB, B, bytes_n, cudaMemcpyHostToDevice);

    

    cudaMemset(d_resR, 0, bytes_n);
    cudaMemset(d_resG, 0, bytes_n);
    cudaMemset(d_resB, 0, bytes_n);
    

    cudaMemcpy(d_kernel, kernel, kbytes_n, cudaMemcpyHostToDevice);


    //Calling function

    dotProduct<<<blockDim,threadsPerBlock>>>(d_imgR, d_kernel, d_resR, height,width);
    dotProduct<<<blockDim,threadsPerBlock>>>(d_imgG, d_kernel, d_resG, height,width);
    dotProduct<<<blockDim,threadsPerBlock>>>(d_imgB, d_kernel, d_resB, height,width);

    cudaDeviceSynchronize();
    // cudaError_t error = cudaGetLastError();
    // if(error!=cudaSuccess){
    //     fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
    //     exit(-1);
    // }

    //Just in case cudaDeviceSynchronize();

    //Copy result back to CPU
    cudaMemcpy(resultR, d_resR, bytes_n, cudaMemcpyDeviceToHost);
    cudaMemcpy(resultG, d_resG, bytes_n, cudaMemcpyDeviceToHost);
    cudaMemcpy(resultB, d_resB, bytes_n, cudaMemcpyDeviceToHost);






    int maxR = 255, maxG = 255, maxB = 255;
    int minR = 0, minG = 0, minB = 0;

    for(int i = 0; i < img.rows; i++ ) 
    { 
        for(int j = 0; j < img.cols; j++ )  
        {
            minR = min(resultR[i*img.cols + j] , minR);   maxR = max(resultR[i*img.cols + j], maxR);
            minG = min(resultG[i*img.cols + j] , minG);   maxG = max(resultG[i*img.cols + j], maxG);
            minB = min(resultB[i*img.cols + j] , minB);   maxB = max(resultB[i*img.cols + j], maxB);
        } 
    }

    

    printf("MaxR %d, minR  %d\n", maxR, minR);
    printf("MaxG %d, minG  %d\n", maxG, minG);
    printf("MaxB %d, minB  %d\n", maxB, minB);
    
    for(int i = 0; i < img.rows; i++ ) 
    { 
        for(int j = 0; j < img.cols; j++ ) 
        {
            //default open cv is BGR
            resultImage.at<cv::Vec3b>(i,j)[2] = float(resultR[i*img.cols + j] - minR) / (maxR - minR) * 255;
            resultImage.at<cv::Vec3b>(i,j)[1] = float(resultG[i*img.cols + j] - minG) / (maxG - minG) * 255;
            resultImage.at<cv::Vec3b>(i,j)[0] = float(resultB[i*img.cols + j] - minB) / (maxB - minB) * 255;

        } 
    }

     

     

    cv::imwrite("out/test.jpg",resultImage);
    // printf("%d\n", resultG[21][213]);
    // printf("max r:%d \n", maxR);

    //Free memory of gpu

    cudaFree(d_imgB);
    cudaFree(d_imgR);
    cudaFree(d_imgG);
    
    cudaFree(d_resR);
    cudaFree(d_resG);
    cudaFree(d_resB);

    cudaFree(d_kernel);
    
    // free memory of cpu
    

    delete [] R;
    delete [] G;
    delete [] B;  
    
    delete [] resultR;
    delete [] resultG;
    delete [] resultB;



    return 0;
}