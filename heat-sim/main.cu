#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cv;

const char* window_name = "Heat simulation";
const int width = 500,
          height = 500;

size_t temps_size = width * height * sizeof(float),
    img_size = width * height * 3 * sizeof(unsigned char);
float* h_temps = new float[temps_size];

__device__ float color_steps[] = {0.0f, 0.25f, 0.50f, 0.75f, 1.0f};
__device__ unsigned char colors[][3] = {
    {0, 0, 0},       // black
    {148, 0, 211},   // violet
    {255, 0, 0},     // red
    {255, 165, 0},   // orange
    {255, 255, 255}  // white
};

int brush_size = 10,
    brush_temp = 100;
bool is_drawing = false;


__device__ void heatmap_color(float val, unsigned char *r, unsigned char *g, unsigned char *b) {
    for (int i = 0;i < 4;i++) {
        if (color_steps[i] <= val && val <= color_steps[i+1]) {
            float blend_factor = (val - color_steps[i]) / (color_steps[i+1] - color_steps[i]);

            *r = (unsigned char)(colors[i][0] + blend_factor * (colors[i+1][0] - colors[i][0]));
            *g = (unsigned char)(colors[i][1] + blend_factor * (colors[i+1][1] - colors[i][1]));
            *b = (unsigned char)(colors[i][2] + blend_factor * (colors[i+1][2] - colors[i][2]));
            return;
        }
    }

    *r = *g = *b = 255;
}

__global__ void calc(float* temps, float* orig_temps, unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        float prev = orig_temps[idx];

        float alpha = 800.0f, // thermal diffusivity
            dx = 1.0f,
            dy = 1.0f,
            dt = 0.000175f, // negative correlation with alpha
            bottom = orig_temps[(y+1) * width + x],
            top = orig_temps[(y-1) * width + x],
            left = orig_temps[y * width + x-1],
            right = orig_temps[y * width + x+1];

        temps[idx] = prev + alpha * dt * (
            (bottom - 2 * prev + top) / (dx * dx) +
            (right - 2 * prev + left) / (dy * dy)
        );

        unsigned char r, g, b;
        heatmap_color(temps[idx], &r, &g, &b);

        img[idx * 3] = b;
        img[idx * 3 + 1] = g;
        img[idx * 3 + 2] = r;
    }
}

void mouse_callback(int event, int x, int y, int flags, void* userdata) {
    bool should_draw = false;
    if (event == EVENT_LBUTTONDOWN) {
        is_drawing = true;
        should_draw = true;
    } else if (event == EVENT_MOUSEMOVE && is_drawing) {
        should_draw = true;
    } else if (event == EVENT_LBUTTONUP) {
        is_drawing = false;
    }

    if (should_draw) {
        for (int i = max(0, x - brush_size);i <= min(width, x + brush_size);i++) {
            for (int j = max(0, y - brush_size);j <= min(height, y + brush_size);j++) {
                int idx = j * width + i;
                h_temps[idx] = brush_temp / 100.0f;
            }
        }
    }
}

int main() {
    memset(h_temps, 0.0f, temps_size);

    for (int y = 225; y < height - 225; y++) {
        for (int x = 225; x < width - 225; x++) {
            h_temps[y * width + x] = 1.0f;
        }
    }

    float *d_temps, *d_temps_copy;
    unsigned char *d_img;
    
    cudaMalloc((void**)&d_temps, temps_size);
    cudaMalloc((void**)&d_temps_copy, temps_size);
    cudaMemcpy(d_temps, h_temps, temps_size, cudaMemcpyHostToDevice);    
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    unsigned char* h_img = new unsigned char[img_size];
    cudaMalloc((void**)&d_img, img_size);

    namedWindow(window_name);
    setMouseCallback(window_name, mouse_callback);
    createTrackbar("Brush size", window_name, &brush_size, 50, NULL);
    createTrackbar("Brush temp", window_name, &brush_temp, 100, NULL);

    double t_start = getTickCount(),
        fps = 0.0f;
    int frameCount = 0;
    
    while (true) {
        cudaMemcpy(d_temps, h_temps, temps_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_temps_copy, d_temps, temps_size, cudaMemcpyDeviceToDevice);
        calc<<<grid, block>>>(d_temps, d_temps_copy, d_img, width, height);
        
        cudaMemcpy(h_temps, d_temps, temps_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        Mat img(height, width, CV_8UC3, h_img);

        char fps_text[20];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.0f", fps);
        putText(img, fps_text, Point(width - 60, 20), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 175, 0), 1);

        imshow(window_name, img);

        frameCount++;
        double t_end = getTickCount();
        double elapsed = (t_end - t_start) / getTickFrequency();

        if (elapsed >= 1.0) {
            fps = frameCount / elapsed;
            t_start = t_end;
            frameCount = 0;
        }

        if ((char)waitKey(1) == 'q') break;
    }
    
    
    delete[] h_temps;
    delete[] h_img;
    cudaFree(d_temps);
    cudaFree(d_temps_copy);
    
    return 0;
}

