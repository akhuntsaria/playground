#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cv;

float color_stops[] = {0.0f, 0.25f, 0.50f, 0.75f, 1.0f};
unsigned char colors[][3] = {
    {0, 0, 0},       // black
    {148, 0, 211},   // violet
    {255, 0, 0},     // red
    {255, 165, 0},   // orange
    {255, 255, 255}  // white
};

__global__ void calc(float* buffer, float* orig, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int index = y * width + x;
        float prev = orig[index];

        float alpha = 23.0f, // thermal diffusivity of iron
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

void heatmap_color(float val, unsigned char *r, unsigned char *g, unsigned char *b) {
    for (int i = 0;i < 4;i++) {
        if (color_stops[i] <= val && val <= color_stops[i+1]) {
            float blend_factor = (val - color_stops[i]) / (color_stops[i+1] - color_stops[i]);

            *r = (unsigned char)(colors[i][0] + blend_factor * (colors[i+1][0] - colors[i][0]));
            *g = (unsigned char)(colors[i][1] + blend_factor * (colors[i+1][1] - colors[i][1]));
            *b = (unsigned char)(colors[i][2] + blend_factor * (colors[i+1][2] - colors[i][2]));
            return;
        }
    }

    *r = *g = *b = 255;
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

    // full spectrum test
    // for (int y = 0; y < 500; y++) {
    //     for (int x = 0; x < 500; x++) {
    //         h_buffer[y * width + x] = y / 500.0f;
    //     }
    // }

    float *d_buffer, *d_buffer_copy;
    
    cudaMalloc((void**)&d_buffer, buffer_size);
    cudaMalloc((void**)&d_buffer_copy, buffer_size);
    cudaMemcpy(d_buffer, h_buffer, buffer_size, cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    unsigned char* h_img = new unsigned char[width * height * 3 * sizeof(unsigned char)];
    
    int iter = 0;
    while (true) {
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
                int idx = j * width + i;

                unsigned char r, g, b;
                heatmap_color(h_buffer[idx], &r, &g, &b);

                h_img[idx*3] = b;
                h_img[idx*3+1] = g;
                h_img[idx*3+2] = r;
            }
        }
        Mat img(height, width, CV_8UC3, h_img);
        imshow("Heat simulation", img);

        if ((char)waitKey(1) == 'q') break;

        if (++iter % 1000 == 0) printf("Iteration %d\n", iter);
    }
    
    
    delete[] h_buffer;
    delete[] h_img;
    cudaFree(d_buffer);
    cudaFree(d_buffer_copy);
    
    return 0;
}

