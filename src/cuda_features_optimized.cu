#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include "cuda_memory_pool.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

struct Point2f {
    float x, y;
    __host__ __device__ Point2f() : x(0), y(0) {}
    __host__ __device__ Point2f(float x_, float y_) : x(x_), y(y_) {}
};

// 全局设备内存缓冲区（用于特征点检测）
static float *d_gradients_Ix = nullptr;
static float *d_gradients_Iy = nullptr;
static float *d_response = nullptr;
static Point2f *d_corners = nullptr;
static int *d_corner_count = nullptr;
static unsigned char *d_valid_corners = nullptr;
static int allocated_image_size = 0;
static int allocated_max_corners = 0;



// 清理特征检测的GPU内存
void cleanup_feature_detection_memory() {
    if (d_gradients_Ix) { cudaFree(d_gradients_Ix); d_gradients_Ix = nullptr; }
    if (d_gradients_Iy) { cudaFree(d_gradients_Iy); d_gradients_Iy = nullptr; }
    if (d_response) { cudaFree(d_response); d_response = nullptr; }
    if (d_corners) { cudaFree(d_corners); d_corners = nullptr; }
    if (d_corner_count) { cudaFree(d_corner_count); d_corner_count = nullptr; }
    if (d_valid_corners) { cudaFree(d_valid_corners); d_valid_corners = nullptr; }
    allocated_image_size = 0;
    allocated_max_corners = 0;
}
// 初始化特征检测的GPU内存
bool init_feature_detection_memory(int image_width, int image_height, int max_corners) {
    int image_size = image_width * image_height;
    
    // 检查是否需要重新分配
    if (image_size <= allocated_image_size && max_corners <= allocated_max_corners) {
        return true; // 已有足够的内存
    }
    
    // 清理旧内存
    cleanup_feature_detection_memory();
    
    // 分配新内存
    CUDA_CHECK(cudaMalloc(&d_gradients_Ix, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradients_Iy, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_response, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corners, max_corners * sizeof(Point2f)));
    CUDA_CHECK(cudaMalloc(&d_corner_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valid_corners, max_corners * sizeof(unsigned char)));
    
    allocated_image_size = image_size;
    allocated_max_corners = max_corners;
    
    return true;
}
// CUDA kernel for computing image gradients (Sobel operators)
__global__ void computeGradients(const float* image, int width, int height,
                                float* Ix, float* Iy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 1 || y < 1 || x >= width-1 || y >= height-1) return;
    
    int idx = y * width + x;
    
    // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
    float gx = -1.0f * image[(y-1)*width + (x-1)] + 1.0f * image[(y-1)*width + (x+1)]
              -2.0f * image[y*width + (x-1)]     + 2.0f * image[y*width + (x+1)]
              -1.0f * image[(y+1)*width + (x-1)] + 1.0f * image[(y+1)*width + (x+1)];
    
    // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
    float gy = -1.0f * image[(y-1)*width + (x-1)] - 2.0f * image[(y-1)*width + x] - 1.0f * image[(y-1)*width + (x+1)]
              +1.0f * image[(y+1)*width + (x-1)] + 2.0f * image[(y+1)*width + x] + 1.0f * image[(y+1)*width + (x+1)];
    
    Ix[idx] = gx / 8.0f;  // Normalize
    Iy[idx] = gy / 8.0f;
}

// CUDA kernel for computing Harris corner response
__global__ void computeHarrisResponse(const float* Ix, const float* Iy, int width, int height,
                                     float* response, float k = 0.04f, int window_size = 3) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_window = window_size / 2;
    if (x < half_window || y < half_window || 
        x >= width - half_window || y >= height - half_window) return;
    
    // Compute elements of the structure tensor M = [Ixx Ixy; Ixy Iyy]
    float Ixx = 0, Iyy = 0, Ixy = 0;
    
    for (int dy = -half_window; dy <= half_window; dy++) {
        for (int dx = -half_window; dx <= half_window; dx++) {
            int idx = (y + dy) * width + (x + dx);
            float ix = Ix[idx];
            float iy = Iy[idx];
            
            Ixx += ix * ix;
            Iyy += iy * iy;
            Ixy += ix * iy;
        }
    }
    
    // Harris corner response: R = det(M) - k * trace(M)^2
    float det = Ixx * Iyy - Ixy * Ixy;
    float trace = Ixx + Iyy;
    response[y * width + x] = det - k * trace * trace;
}

// CUDA kernel for non-maximum suppression
__global__ void nonMaxSuppression(const float* response, int width, int height,
                                 Point2f* corners, int* corner_count, int max_corners,
                                 float quality_level, float min_distance) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 2 || y < 2 || x >= width-2 || y >= height-2) return;
    
    int idx = y * width + x;
    float center_response = response[idx];
    
    // Check if this is a local maximum and above threshold
    if (center_response < quality_level) return;
    
    // Check 3x3 neighborhood for local maximum
    bool is_maximum = true;
    for (int dy = -1; dy <= 1 && is_maximum; dy++) {
        for (int dx = -1; dx <= 1 && is_maximum; dx++) {
            if (dx == 0 && dy == 0) continue;
            int neighbor_idx = (y + dy) * width + (x + dx);
            if (response[neighbor_idx] > center_response) {
                is_maximum = false;
            }
        }
    }
    
    if (!is_maximum) return;
    
    // Add corner if it passes minimum distance check
    int corner_idx = atomicAdd(corner_count, 1);
    if (corner_idx < max_corners) {
        corners[corner_idx] = Point2f(x, y);
    }
}

// CUDA kernel for enforcing minimum distance between corners
__global__ void enforceMinDistance(Point2f* corners, int num_corners, 
                                  float min_distance, unsigned char* valid_corners) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_corners) return;
    
    valid_corners[tid] = 1;  // true
    Point2f current = corners[tid];
    
    // Check against all previous corners
    for (int i = 0; i < tid; i++) {
        if (!valid_corners[i]) continue;
        
        Point2f other = corners[i];
        float dx = current.x - other.x;
        float dy = current.y - other.y;
        float distance = sqrtf(dx * dx + dy * dy);
        
        if (distance < min_distance) {
            valid_corners[tid] = 0;  // false
            return;
        }
    }
}

// 优化后的特征点检测函数 - 使用预分配的内存
std::vector<cv::Point2f> cuda_goodFeaturesToTrack_optimized(const cv::Mat& image, 
                                                           CudaMemoryPool* memory_pool,
                                                           int max_corners,
                                                           double quality_level,
                                                           double min_distance) {
    
    // Convert image to grayscale float if needed
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image;
    }
    
    cv::Mat float_image;
    gray_image.convertTo(float_image, CV_32F, 1.0/255.0);
    
    int width = float_image.cols;
    int height = float_image.rows;
    
    // 初始化特征检测内存
    if (!init_feature_detection_memory(width, height, max_corners)) {
        return std::vector<cv::Point2f>();
    }
    
    // 检查内存池是否已初始化且图像尺寸是否在范围内
    if (!memory_pool->isInitialized()) {
        std::cerr << "Memory pool not initialized" << std::endl;
        return std::vector<cv::Point2f>();
    }
    
    if (width * height > memory_pool->getImageSize()) {
        std::cerr << "Image size (" << width << "x" << height 
                  << ") exceeds memory pool capacity (" << memory_pool->getImageSize() << ")" << std::endl;
        return std::vector<cv::Point2f>();
    }
    
    // 使用内存池的图像缓冲区
    float* d_image = memory_pool->getDeviceImageBuffer();
    
    // Copy image data to GPU using the memory pool's stream
    CUDA_CHECK(cudaMemcpyAsync(d_image, float_image.ptr<float>(0), 
                              width * height * sizeof(float), 
                              cudaMemcpyHostToDevice, memory_pool->stream_));
    
    // Initialize corner count
    int zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(d_corner_count, &zero, sizeof(int), 
                              cudaMemcpyHostToDevice, memory_pool->stream_));
    
    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Step 1: Compute gradients
    computeGradients<<<grid, block, 0, memory_pool->stream_>>>(
        d_image, width, height, d_gradients_Ix, d_gradients_Iy);
    
    // Step 2: Compute Harris corner response
    computeHarrisResponse<<<grid, block, 0, memory_pool->stream_>>>(
        d_gradients_Ix, d_gradients_Iy, width, height, d_response);
    
    // Step 3: Find local maxima and apply quality threshold
    nonMaxSuppression<<<grid, block, 0, memory_pool->stream_>>>(
        d_response, width, height, d_corners, d_corner_count, 
        max_corners, quality_level, min_distance);
    
    // Get number of detected corners
    int num_corners;
    CUDA_CHECK(cudaMemcpyAsync(&num_corners, d_corner_count, sizeof(int), 
                              cudaMemcpyDeviceToHost, memory_pool->stream_));
    CUDA_CHECK(cudaStreamSynchronize(memory_pool->stream_));
    
    if (num_corners == 0) {
        return std::vector<cv::Point2f>();
    }
    
    // Limit to max_corners
    num_corners = std::min(num_corners, max_corners);
    
    // Step 4: Enforce minimum distance constraint
    dim3 distance_block(256);
    dim3 distance_grid((num_corners + distance_block.x - 1) / distance_block.x);
    
    enforceMinDistance<<<distance_grid, distance_block, 0, memory_pool->stream_>>>(
        d_corners, num_corners, min_distance, d_valid_corners);
    
    // Copy results back to host
    std::vector<Point2f> h_corners(num_corners);
    std::vector<unsigned char> h_valid(num_corners);
    
    CUDA_CHECK(cudaMemcpyAsync(h_corners.data(), d_corners, 
                              num_corners * sizeof(Point2f), 
                              cudaMemcpyDeviceToHost, memory_pool->stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_valid.data(), d_valid_corners, 
                              num_corners * sizeof(unsigned char), 
                              cudaMemcpyDeviceToHost, memory_pool->stream_));
    CUDA_CHECK(cudaStreamSynchronize(memory_pool->stream_));
    
    // Filter valid corners and convert to OpenCV format
    std::vector<cv::Point2f> result;
    result.reserve(num_corners);
    for (int i = 0; i < num_corners; i++) {
        if (h_valid[i]) {
            result.push_back(cv::Point2f(h_corners[i].x, h_corners[i].y));
        }
    }
    
    return result;
}

// 清理函数，在程序结束时调用
void cleanup_cuda_features() {
    cleanup_feature_detection_memory();
}
