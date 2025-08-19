#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

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

// Host function for CUDA-based goodFeaturesToTrack
std::vector<cv::Point2f> cuda_goodFeaturesToTrack(const cv::Mat& image, 
                                                  int max_corners = 800,
                                                  double quality_level = 0.02,
                                                  double min_distance = 10.0) {
    
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
    
    // Allocate GPU memory
    float *d_image, *d_Ix, *d_Iy, *d_response;
    Point2f *d_corners;
    int *d_corner_count;
    unsigned char *d_valid_corners;  // 改为unsigned char
    
    CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ix, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Iy, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_response, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corners, max_corners * sizeof(Point2f)));
    CUDA_CHECK(cudaMalloc(&d_corner_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valid_corners, max_corners * sizeof(unsigned char)));
    
    // Copy image data to GPU
    CUDA_CHECK(cudaMemcpy(d_image, float_image.ptr<float>(0), 
                         width * height * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize corner count
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_corner_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Step 1: Compute gradients
    computeGradients<<<grid, block>>>(d_image, width, height, d_Ix, d_Iy);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 2: Compute Harris corner response
    computeHarrisResponse<<<grid, block>>>(d_Ix, d_Iy, width, height, d_response);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 3: Find local maxima and apply quality threshold
    nonMaxSuppression<<<grid, block>>>(d_response, width, height, d_corners, d_corner_count, 
                                      max_corners, quality_level, min_distance);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get number of detected corners
    int num_corners;
    CUDA_CHECK(cudaMemcpy(&num_corners, d_corner_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (num_corners == 0) {
        // Cleanup and return empty vector
        cudaFree(d_image);
        cudaFree(d_Ix);
        cudaFree(d_Iy);
        cudaFree(d_response);
        cudaFree(d_corners);
        cudaFree(d_corner_count);
        cudaFree(d_valid_corners);
        return std::vector<cv::Point2f>();
    }
    
    // Limit to max_corners
    num_corners = std::min(num_corners, max_corners);
    
    // Step 4: Enforce minimum distance constraint
    dim3 distance_block(256);
    dim3 distance_grid((num_corners + distance_block.x - 1) / distance_block.x);
    
    enforceMinDistance<<<distance_grid, distance_block>>>(d_corners, num_corners, 
                                                         min_distance, d_valid_corners);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    std::vector<Point2f> h_corners(num_corners);
    std::vector<unsigned char> h_valid(num_corners);  // 使用unsigned char而不是bool
    
    CUDA_CHECK(cudaMemcpy(h_corners.data(), d_corners, 
                         num_corners * sizeof(Point2f), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid.data(), d_valid_corners, 
                         num_corners * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    // Filter valid corners and convert to OpenCV format
    std::vector<cv::Point2f> result;
    for (int i = 0; i < num_corners; i++) {
        if (h_valid[i]) {
            result.push_back(cv::Point2f(h_corners[i].x, h_corners[i].y));
        }
    }
    
    // Cleanup GPU memory
    cudaFree(d_image);
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_response);
    cudaFree(d_corners);
    cudaFree(d_corner_count);
    cudaFree(d_valid_corners);
    
    return result;
}

// CUDA kernel for optical flow using Lucas-Kanade method
__global__ void cudaOpticalFlowLK(const float* img1, const float* img2, 
                                 int width, int height,
                                 const Point2f* pts1, Point2f* pts2, 
                                 unsigned char* status, float* err,
                                 int num_points, int window_size = 5) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;
    
    Point2f p1 = pts1[tid];
    int x = (int)p1.x, y = (int)p1.y;
    int half_window = window_size / 2;
    
    // Check bounds
    if (x < half_window || y < half_window || 
        x >= width - half_window || y >= height - half_window) {
        status[tid] = 0;
        return;
    }
    
    // Compute gradients and temporal difference in the window
    float A11 = 0, A12 = 0, A22 = 0;  // Elements of ATA matrix
    float b1 = 0, b2 = 0;              // Elements of ATb vector
    
    for (int dy = -half_window; dy <= half_window; dy++) {
        for (int dx = -half_window; dx <= half_window; dx++) {
            int px = x + dx, py = y + dy;
            int idx = py * width + px;
            
            // Spatial gradients (simple finite differences)
            float Ix = 0, Iy = 0;
            if (px > 0 && px < width-1) {
                Ix = (img1[idx+1] - img1[idx-1]) * 0.5f;
            }
            if (py > 0 && py < height-1) {
                Iy = (img1[idx+width] - img1[idx-width]) * 0.5f;
            }
            
            // Temporal difference
            float It = img2[idx] - img1[idx];
            
            // Accumulate normal equations
            A11 += Ix * Ix;
            A12 += Ix * Iy;
            A22 += Iy * Iy;
            b1 -= Ix * It;
            b2 -= Iy * It;
        }
    }
    
    // Solve 2x2 system: [A11 A12; A12 A22] * [u; v] = [b1; b2]
    float det = A11 * A22 - A12 * A12;
    
    if (fabsf(det) < 1e-6f) {
        status[tid] = 0;
        return;
    }
    
    float u = (A22 * b1 - A12 * b2) / det;
    float v = (A11 * b2 - A12 * b1) / det;
    
    pts2[tid] = Point2f(p1.x + u, p1.y + v);
    
    // Compute tracking error (SSD in window)
    float error = 0;
    int valid_pixels = 0;
    
    for (int dy = -half_window; dy <= half_window; dy++) {
        for (int dx = -half_window; dx <= half_window; dx++) {
            int px1 = x + dx, py1 = y + dy;
            int px2 = (int)(p1.x + u + dx), py2 = (int)(p1.y + v + dy);
            
            if (px1 >= 0 && px1 < width && py1 >= 0 && py1 < height &&
                px2 >= 0 && px2 < width && py2 >= 0 && py2 < height) {
                
                float diff = img1[py1 * width + px1] - img2[py2 * width + px2];
                error += diff * diff;
                valid_pixels++;
            }
        }
    }
    
    if (valid_pixels > 0) {
        error /= valid_pixels;
    }
    
    // Set status based on error threshold and displacement magnitude
    float displacement = sqrtf(u * u + v * v);
    status[tid] = (error < 0.01f && displacement < 50.0f) ? 1 : 0;
    err[tid] = error;
}

// Host function for CUDA optical flow
void cuda_calcOpticalFlowPyrLK(const cv::Mat& img1, const cv::Mat& img2,
                              const std::vector<cv::Point2f>& pts1,
                              std::vector<cv::Point2f>& pts2,
                              std::vector<uchar>& status,
                              std::vector<float>& err) {
    
    // Convert images to float
    cv::Mat float_img1, float_img2;
    img1.convertTo(float_img1, CV_32F, 1.0/255.0);
    img2.convertTo(float_img2, CV_32F, 1.0/255.0);
    
    int width = float_img1.cols;
    int height = float_img1.rows;
    int num_points = pts1.size();
    
    // Allocate GPU memory
    float *d_img1, *d_img2;
    Point2f *d_pts1, *d_pts2;
    unsigned char *d_status;
    float *d_err;
    
    CUDA_CHECK(cudaMalloc(&d_img1, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pts1, num_points * sizeof(Point2f)));
    CUDA_CHECK(cudaMalloc(&d_pts2, num_points * sizeof(Point2f)));
    CUDA_CHECK(cudaMalloc(&d_status, num_points * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_err, num_points * sizeof(float)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_img1, float_img1.ptr<float>(0), 
                         width * height * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img2, float_img2.ptr<float>(0), 
                         width * height * sizeof(float), cudaMemcpyHostToDevice));
    
    // Convert and copy points
    std::vector<Point2f> cuda_pts1(num_points);
    for (int i = 0; i < num_points; i++) {
        cuda_pts1[i] = Point2f(pts1[i].x, pts1[i].y);
    }
    CUDA_CHECK(cudaMemcpy(d_pts1, cuda_pts1.data(), 
                         num_points * sizeof(Point2f), cudaMemcpyHostToDevice));
    
    // Launch optical flow kernel
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    
    cudaOpticalFlowLK<<<grid, block>>>(d_img1, d_img2, width, height,
                                      d_pts1, d_pts2, d_status, d_err, num_points);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    std::vector<Point2f> cuda_pts2(num_points);
    CUDA_CHECK(cudaMemcpy(cuda_pts2.data(), d_pts2, 
                         num_points * sizeof(Point2f), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(status.data(), d_status, 
                         num_points * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(err.data(), d_err, 
                         num_points * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Convert points back to OpenCV format
    pts2.resize(num_points);
    for (int i = 0; i < num_points; i++) {
        pts2[i] = cv::Point2f(cuda_pts2[i].x, cuda_pts2[i].y);
    }
    
    // Cleanup
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_pts1);
    cudaFree(d_pts2);
    cudaFree(d_status);
    cudaFree(d_err);
}
