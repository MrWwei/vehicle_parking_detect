#ifndef CUDA_MEMORY_POOL_H
#define CUDA_MEMORY_POOL_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

class CudaMemoryPool {
public:
    CudaMemoryPool();
    ~CudaMemoryPool();
    
    bool init(int max_points, int image_width, int image_height);
    void cleanup();
    
    // 获取预分配的设备内存
    float* getDevicePointsBuffer() { return d_points_; }
    float* getDeviceWorldBuffer() { return d_world_; }
    float* getDeviceImageBuffer() { return d_image_; }
    float* getDeviceHomographyBuffer() { return d_homography_; }
    
    // 获取缓冲区大小
    size_t getMaxPoints() const { return max_points_; }
    size_t getImageSize() const { return image_width_ * image_height_; }
    
    // 内存池状态检查
    bool isInitialized() const { return initialized_; }
    cudaStream_t stream_;
    
private:
    bool initialized_;
    int max_points_;
    int image_width_;
    int image_height_;
    
    // 设备内存指针
    float* d_points_;     // 输入点坐标 (max_points * 2)
    float* d_world_;      // 世界坐标输出 (max_points * 2)
    float* d_image_;      // 图像数据 (width * height)
    float* d_homography_; // 齐次变换矩阵 (3 * 3)
public:
    // 主机内存固定内存（pinned memory）
    float* h_points_pinned_;
    float* h_world_pinned_;
    
    // CUDA流
};

// 全局内存池实例
extern std::unique_ptr<CudaMemoryPool> g_cuda_memory_pool;

// 优化后的CUDA函数声明
void launch_world_coord_optimized(const std::vector<cv::Point2f>& pts,
                                 const cv::Mat& H,
                                 std::vector<cv::Point2f>& world,
                                 CudaMemoryPool* memory_pool);

std::vector<cv::Point2f> cuda_goodFeaturesToTrack_optimized(const cv::Mat& image,
                                                           CudaMemoryPool* memory_pool,
                                                           int max_corners = 800,
                                                           double quality_level = 0.02,
                                                           double min_distance = 10.0);

#endif // CUDA_MEMORY_POOL_H
