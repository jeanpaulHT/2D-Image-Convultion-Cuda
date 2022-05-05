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
    
    int x =  blockIdx.y * blockDim.y + threadIdx.x;
    int y =  blockIdx.x * blockDim.x + threadIdx.y;

    if(x >= width || y >= height) return ;
    if(x == 0 || x == width-1 || y == 0 || y == height-1) {res[x * width + y] = 0; return;}


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


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(){
    
    constexpr int width = 2560, height = 2560;
    
    cv::Mat img = cv::imread("in/cow.jpg");
     
    
  
    int (*R)[width] = new int[height][width];
    int (*G)[width] = new int[height][width];
    int (*B)[width] = new int[height][width];
  
    cv::Mat resultImage = cv::Mat::zeros(img.size(), CV_8UC3);
    
    
    for(int j = 0; j<img.rows; j++ ) 
    { 
        for(int i = 0; i<img.cols; i++ ) 
        {
            // the default order for openCV is BGR
            R[j][i]  = img.at<cv::Vec3b>(j,i)[2];
            G[j][i]  = img.at<cv::Vec3b>(j,i)[1];
            B[j][i]  = img.at<cv::Vec3b>(j,i)[0];
        } 
     }

    
    // dimension of the image array, assume that is a square
    // for the calculation of threads
    int N  = std::max(height, width);



    int (*resultR)[width] = new int[height][width];
    int (*resultG)[width] = new int[height][width];
    int (*resultB)[width] = new int[height][width];


    int kernel[KS][KS] = {
        {-1,0,1},
        {-1,0,1},
        {-1,0,1}
    };



    
    // Calculating <numBlocks, Threads per block>
    int tpb = min(16, N); //thread per block
    int numBlocks = (N + tpb - 1) / tpb;

    dim3 blockDim (numBlocks, numBlocks);
    dim3 threadsPerBlock(tpb,tpb);



    // Allocating space in device memory
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
    
    cudaDeviceSynchronize();

    dotProduct<<<blockDim,threadsPerBlock>>>(d_imgG, d_kernel, d_resG, height,width);

    cudaDeviceSynchronize();

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

    for(int j = 0; j<img.rows; j++ ) 
    { 
        for(int i = 0; i<img.cols; i++ ) 
        {
            minR = min(resultR[j][i]  , minR);   maxR = max(resultR[j][i], maxR);
            minG = min(resultR[j][i] , minG);   maxG = max(resultR[j][i], maxG);
            minB = min(resultB[j][i] , minB);   maxG = max(resultB[j][i], maxB);
        } 
    }

    

    printf("MaxR %d, minR  %d\n", maxR, minR);
    printf("MaxG %d, minG  %d\n", maxG, minG);
    printf("MaxB %d, minB  %d\n", maxB, minB);
    
    for(int j = 0; j<img.rows; j++ ) 
    { 
        for(int i = 0; i<img.cols; i++ ) 
        {
            //default open cv is BRG
            resultImage.at<cv::Vec3b>(j,i)[2] = float(resultR[j][i]- minR) / (maxR - minR) * 255;
            resultImage.at<cv::Vec3b>(j,i)[1] = float(resultG[j][i] - minG) / (maxG - minG) * 255;
            resultImage.at<cv::Vec3b>(j,i)[0] = float(resultG[j][i]- minG) / (maxG - minG) * 255;

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