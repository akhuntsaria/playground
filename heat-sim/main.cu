#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cv;

__global__ void calc(unsigned char* buffer, unsigned char* orig, int width, int height, int iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int index = (y * width + x) * 3;
        float prev = buffer[index + 2] / 255.0f; // take from and write to the red channel

        /* u_current[i][j] = u_prev[i][j] + alpha * dt * (
            (u_prev[i+1][j] - 2 * u_prev[i][j] + u_prev[i-1][j]) / (dx * dx) +  # x-direction
            (u_prev[i][j+1] - 2 * u_prev[i][j] + u_prev[i][j-1]) / (dy * dy)   # y-direction
        ) */
        float alpha = 165.0f,
            dx = 640.0f,
            dy = 480.0f,
            dt = min(dx * dx / (4 * alpha), dy * dy / (4 * alpha));

        float bottom = x + 1 < width ? (orig[(y * width + x + 1) * 3 + 2]) : 0.0f,
            top = x - 1 >= 0 ? (orig[(y * width + x - 1) * 3 + 2]) : 0.0f,
            left = y - 1 >= 0 ? (orig[((y-1) * width + x) * 3 + 2]) : 0.0f,
            right = y + 1 < height ? (orig[((y+1) * width + x) * 3 + 2]) : 0.0f;
        
        float curr = prev + alpha * dt * (
            (bottom - 2 * prev + top) / (dx * dx) +
            (right - 2 * prev + left) / (dy * dy)
        );
        buffer[index + 2] = (int)(curr * 255) % 255;
    }
}

int main() {
    const int width = 640;
    const int height = 480;

    unsigned char* h_buffer = new unsigned char[width * height * 3];
    
    /* for (int y = height - 20; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 3;
            h_buffer[index] = 0;
            h_buffer[index + 1] = 0;
            h_buffer[index + 2] = 255;
        }
    } */

    for (int y = 50; y < height - 50; y++) {
        for (int x = 50; x < width - 50; x++) {
            int index = (y * width + x) * 3;
            h_buffer[index] = 0;
            h_buffer[index + 1] = 0;
            h_buffer[index + 2] = 255/2;
        }
    }

    unsigned char *d_buffer, *d_buffer_copy;
    size_t buffer_size = width * height * 3 * sizeof(unsigned char);
    
    cudaMalloc((void**)&d_buffer, buffer_size);
    cudaMalloc((void**)&d_buffer_copy, buffer_size);
    cudaMemcpy(d_buffer, h_buffer, buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer_copy, d_buffer, buffer_size, cudaMemcpyDeviceToDevice);
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // No controls
    //namedWindow("Heat simulation", WINDOW_GUI_NORMAL);

    int iter = 0;
    while (iter < 255) {
        printf("Iteration %d\n", ++iter);

        calc<<<grid, block>>>(d_buffer, d_buffer_copy, width, height, iter);
        
        cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        Mat img(height, width, CV_8UC3, h_buffer);
        imshow("Heat simulation", img);

        if ((char)waitKey(30) == 'q') break;
    }
    
    
    delete[] h_buffer;
    cudaFree(d_buffer);
    
    return 0;
}

