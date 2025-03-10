#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cv;

__global__ void calc(float* buffer, float* orig, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int index = y * width + x;
        float prev = orig[index];

        float alpha = 25.0f,
            dx = 1.0f,
            dy = 1.0f,
            dt = 0.01f,
            bottom = orig[(y+1) * width + x],
            top = orig[(y-1) * width + x],
            left = orig[y * width + x-1],
            right = orig[y * width + x+1];

        buffer[index] = prev + alpha * dt * (
            (bottom - 2 * prev + top) / (dx * dx) +
            (right - 2 * prev + left) / (dy * dy)
        );
    }
}

int main() {
    const int width = 500,
        height = 500;
    size_t buffer_size = width * height * sizeof(float);

    float* h_buffer = new float[buffer_size];
    memset(h_buffer, 0.0f, buffer_size);

    for (int y = 125; y < height - 125; y++) {
        for (int x = 125; x < width - 125; x++) {
            h_buffer[y * width + x] = 1.0f;
        }
    }

    float *d_buffer, *d_buffer_copy;
    
    cudaMalloc((void**)&d_buffer, buffer_size);
    cudaMalloc((void**)&d_buffer_copy, buffer_size);
    cudaMemcpy(d_buffer, h_buffer, buffer_size, cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    unsigned char* h_img = new unsigned char[width * height * 3 * sizeof(unsigned char)];
    
    int iter = 0;
    while (true) {
        printf("Iteration %d\n", ++iter);

        cudaMemcpy(d_buffer_copy, d_buffer, buffer_size, cudaMemcpyDeviceToDevice);
        calc<<<grid, block>>>(d_buffer, d_buffer_copy, width, height);
        
        cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        //TODO in kernel
        for (int i = 0;i < width;i++) {
            for (int j = 0;j < height;j++) {
                int idx = j*width+i;
                float val = h_buffer[idx];
                unsigned char r, g, b;
                if (val < 0.3f) {
                    r = (unsigned char)(val * 255);
                    g = 0;
                    b = 0;
                } else if (val < 0.7f) {
                    r = 255;
                    g = (unsigned char)((val - 0.3f) * (255 / 0.4f));
                    b = 0;
                } else {
                    r = 255;
                    g = 255;
                    b = (unsigned char)((val - 0.7f) * (255 / 0.3f));
                }
                h_img[idx*3] = b;
                h_img[idx*3+1] = g;
                h_img[idx*3+2] = r;
            }
        }
        Mat img(height, width, CV_8UC3, h_img);
        imshow("Heat simulation", img);

        if ((char)waitKey(1) == 'q') break;
    }
    
    
    delete[] h_buffer;
    delete[] h_img;
    cudaFree(d_buffer);
    cudaFree(d_buffer_copy);
    
    return 0;
}

