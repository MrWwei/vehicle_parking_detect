#include "../include/cuda_memory_pool.h"
#include <iostream>

std::unique_ptr<CudaMemoryPool> g_cuda_memory_pool = nullptr;

CudaMemoryPool::CudaMemoryPool() 
    : initialized_(false), max_points_(0), image_width_(0), image_height_(0),
      d_points_(nullptr), d_world_(nullptr), d_image_(nullptr), d_homography_(nullptr),
      h_points_pinned_(nullptr), h_world_pinned_(nullptr), stream_(0) {
}

CudaMemoryPool::~CudaMemoryPool() {
    cleanup();
}

bool CudaMemoryPool::init(int max_points, int image_width, int image_height) {
    if (initialized_) {
        cleanup();
    }
    
    max_points_ = max_points;
    image_width_ = image_width;
    image_height_ = image_height;
    
    cudaError_t error;
    
    // 创建CUDA流
    error = cudaStreamCreate(&stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // 分配设备内存
    error = cudaMalloc(&d_points_, max_points * 2 * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for points: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_world_, max_points * 2 * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for world coords: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_image_, image_width * image_height * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for image: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_homography_, 9 * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for homography: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // 分配固定主机内存（pinned memory）以提高传输效率
    error = cudaMallocHost(&h_points_pinned_, max_points * 2 * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned host memory for points: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMallocHost(&h_world_pinned_, max_points * 2 * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned host memory for world coords: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "CUDA memory pool initialized successfully:" << std::endl;
    std::cout << "  Max points: " << max_points << std::endl;
    std::cout << "  Image size: " << image_width << "x" << image_height << std::endl;
    std::cout << "  Total GPU memory allocated: " << 
                 (max_points * 4 + image_width * image_height + 9) * sizeof(float) / (1024*1024) << " MB" << std::endl;
    
    return true;
}

void CudaMemoryPool::cleanup() {
    if (!initialized_) return;
    
    // 释放设备内存
    if (d_points_) { cudaFree(d_points_); d_points_ = nullptr; }
    if (d_world_) { cudaFree(d_world_); d_world_ = nullptr; }
    if (d_image_) { cudaFree(d_image_); d_image_ = nullptr; }
    if (d_homography_) { cudaFree(d_homography_); d_homography_ = nullptr; }
    
    // 释放固定主机内存
    if (h_points_pinned_) { cudaFreeHost(h_points_pinned_); h_points_pinned_ = nullptr; }
    if (h_world_pinned_) { cudaFreeHost(h_world_pinned_); h_world_pinned_ = nullptr; }
    
    // 销毁流
    if (stream_) { cudaStreamDestroy(stream_); stream_ = 0; }
    
    initialized_ = false;
    std::cout << "CUDA memory pool cleaned up" << std::endl;
}

// CUDA核函数：世界坐标变换
__global__ void world_coord_transform_kernel(const float* points, const float* homography, 
                                            float* world_coords, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float x = points[idx * 2];
    float y = points[idx * 2 + 1];
    
    // 齐次坐标变换：[x', y', w'] = H * [x, y, 1]
    float x_prime = homography[0] * x + homography[1] * y + homography[2];
    float y_prime = homography[3] * x + homography[4] * y + homography[5];
    float w_prime = homography[6] * x + homography[7] * y + homography[8];
    
    // 归一化
    if (w_prime != 0.0f) {
        world_coords[idx * 2] = x_prime / w_prime;
        world_coords[idx * 2 + 1] = y_prime / w_prime;
    } else {
        world_coords[idx * 2] = x;
        world_coords[idx * 2 + 1] = y;
    }
}

void launch_world_coord_optimized(const std::vector<cv::Point2f>& pts,
                                 const cv::Mat& H,
                                 std::vector<cv::Point2f>& world,
                                 CudaMemoryPool* memory_pool) {
    if (!memory_pool || !memory_pool->isInitialized() || pts.empty()) {
        return;
    }
    
    int num_points = pts.size();
    if (num_points > memory_pool->getMaxPoints()) {
        std::cerr << "Number of points exceeds memory pool capacity" << std::endl;
        return;
    }
    
    // 准备主机数据
    float* h_points = memory_pool->h_points_pinned_;
    float* h_world = memory_pool->h_world_pinned_;
    
    // 复制点坐标到固定内存
    for (int i = 0; i < num_points; ++i) {
        h_points[i * 2] = pts[i].x;
        h_points[i * 2 + 1] = pts[i].y;
    }
    
    // 复制齐次变换矩阵
    float h_homography[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            h_homography[i * 3 + j] = H.at<double>(i, j);
        }
    }
    
    // 异步复制到设备内存
    cudaMemcpyAsync(memory_pool->getDevicePointsBuffer(), h_points, 
                   num_points * 2 * sizeof(float), cudaMemcpyHostToDevice, memory_pool->stream_);
    cudaMemcpyAsync(memory_pool->getDeviceHomographyBuffer(), h_homography, 
                   9 * sizeof(float), cudaMemcpyHostToDevice, memory_pool->stream_);
    
    // 配置CUDA核函数
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;
    
    // 启动核函数
    world_coord_transform_kernel<<<grid_size, block_size, 0, memory_pool->stream_>>>(
        memory_pool->getDevicePointsBuffer(),
        memory_pool->getDeviceHomographyBuffer(),
        memory_pool->getDeviceWorldBuffer(),
        num_points
    );
    
    // 异步复制结果回主机
    cudaMemcpyAsync(h_world, memory_pool->getDeviceWorldBuffer(), 
                   num_points * 2 * sizeof(float), cudaMemcpyDeviceToHost, memory_pool->stream_);
    
    // 同步等待完成
    cudaStreamSynchronize(memory_pool->stream_);
    
    // 复制结果到输出向量
    world.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
        world[i].x = h_world[i * 2];
        world[i].y = h_world[i * 2 + 1];
    }
}

// std::vector<cv::Point2f> cuda_goodFeaturesToTrack_optimized(const cv::Mat& image,
//                                                            CudaMemoryPool* memory_pool,
//                                                            int max_corners,
//                                                            double quality_level,
//                                                            double min_distance) {
//     // 这里暂时使用CPU版本，可以进一步优化为GPU版本
//     std::vector<cv::Point2f> corners;
//     cv::goodFeaturesToTrack(image, corners, max_corners, quality_level, min_distance);
//     return corners;
// }
