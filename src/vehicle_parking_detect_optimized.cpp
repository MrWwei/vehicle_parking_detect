#include "vehicle_parking_detect.h"
#include "cuda_memory_pool.h"
#include <chrono>
#include <iostream>

// 声明优化后的CUDA函数
// void launch_world_coord_optimized(const std::vector<cv::Point2f>& pts,
//                                  const cv::Mat& H,
//                                  std::vector<cv::Point2f>& world,
//                                  CudaMemoryPool* memory_pool);

// std::vector<cv::Point2f> cuda_goodFeaturesToTrack_optimized(const cv::Mat& image,
//                                                            CudaMemoryPool* memory_pool,
//                                                            int max_corners,
//                                                            double quality_level,
//                                                            double min_distance);

class VehicleParkingDetectOptimized : public VehicleParkingDetect {
public:
    VehicleParkingDetectOptimized() {
        memory_pool_.reset(new CudaMemoryPool());
    }
    
    
    ~VehicleParkingDetectOptimized() override {
        if (memory_pool_) {
            memory_pool_->cleanup();
        }
    }
    
    bool init(VehicleParkingInitParams& params) override {
        params_ = params;
        
        // 初始化CUDA内存池
        // 预估最大图像尺寸和最大特征点数量
        int max_image_width = 3840;   // 可根据实际需求调整
        int max_image_height = 2160;
        int max_features = params_.MAX_FEATURES;
        
        if (!memory_pool_->init(max_features, max_image_width, max_image_height)) {
            std::cerr << "Failed to initialize CUDA memory pool" << std::endl;
            return false;
        }
        
        // 预分配临时缓冲区
        temp_gray_.create(max_image_height, max_image_width, CV_8UC1);
        temp_prev_gray_.create(max_image_height, max_image_width, CV_8UC1);
        
        std::cout << "VehicleParkingDetectOptimized initialized successfully" << std::endl;
        return true;
    }
    
    bool detect(const cv::Mat& gray_mat, std::vector<TrackBox>& tracks) override {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 避免不必要的图像拷贝
        cv::Mat gray;
        if (gray_mat.channels() != 1) {
            cv::cvtColor(gray_mat, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = gray_mat;  // 直接引用，避免拷贝
        }
        
        // 使用优化后的同态变换计算
        cv::Mat H_curr;
        if (prev_gray_.empty()) {
            H_curr = cv::Mat::eye(3, 3, CV_64F);
            prev_gray_ = gray.clone();  // 第一帧必须拷贝
        } else {
            H_curr = get_homography_optimized(prev_gray_, gray);
            // 使用浅拷贝策略
            std::swap(prev_gray_, temp_prev_gray_);
            gray.copyTo(prev_gray_);
        }
        
        ref_H_acc_ = H_curr * ref_H_acc_;
        
        // 重置参考帧
        if (frame_id_ % params_.RESET_EVERY == 0) {
            ref_H_acc_ = cv::Mat::eye(3, 3, CV_64F);
            world_hist_.clear();
            still_count_.clear();
        }
        
        // 准备坐标 - 预分配vector大小
        if (input_points_.size() != tracks.size()) {
            input_points_.resize(tracks.size());
        }
        
        for (size_t i = 0; i < tracks.size(); ++i) {
            int center_x = tracks[i].box.x + tracks[i].box.width / 2;
            int center_y = tracks[i].box.y + tracks[i].box.height / 2;
            input_points_[i] = cv::Point2f(center_x, center_y);
        }
        
        // 使用优化后的CUDA世界坐标变换
        world_coords_.clear();
        if (!input_points_.empty()) {
            launch_world_coord_optimized(input_points_, ref_H_acc_, world_coords_, memory_pool_.get());
        }
        
        // 处理跟踪结果
        for (size_t i = 0; i < tracks.size(); ++i) {
            auto& track_box = tracks[i];
            int id = track_box.track_id;
            
            if (i < world_coords_.size()) {
                // 使用预分配的deque，避免频繁内存分配
                auto& hist = world_hist_[id];
                hist.push_back(world_coords_[i]);
                if (static_cast<int>(hist.size()) > params_.K) {
                    hist.pop_front();
                }
                
                bool is_still = false;
                double delta = 0;
                if (hist.size() >= 2) {
                    double sum = 0;
                    const auto& first_point = hist[0];
                    for (size_t j = 1; j < hist.size(); ++j) {
                        sum += cv::norm(hist[j] - first_point);
                    }
                    delta = sum / (hist.size() - 1);
                    is_still = delta < params_.EPS_WORLD;
                    track_box.delta = delta;
                }
                
                still_count_[id] = is_still ? still_count_[id] + 1 : 0;
                track_box.is_still = still_count_[id] >= params_.MIN_SPEED_FRAMES;
            }
        }
        
        frame_id_++;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // 可选的性能统计
        if (frame_id_ % 100 == 0) {
            std::cout << "Frame " << frame_id_ << " processing time: " 
                      << duration.count() / 1000.0 << " ms" << std::endl;
        }
        
        return true;
    }
    
private:
    VehicleParkingInitParams params_;
    std::unique_ptr<CudaMemoryPool> memory_pool_;
    
    // 预分配的缓冲区
    cv::Mat temp_gray_;
    cv::Mat temp_prev_gray_;
    std::vector<cv::Point2f> input_points_;
    std::vector<cv::Point2f> world_coords_;
    
    // 算法状态
    std::unordered_map<int, std::deque<cv::Point2f>> world_hist_;
    std::unordered_map<int, int> still_count_;
    cv::Mat prev_gray_;
    cv::Mat ref_H_acc_ = cv::Mat::eye(3, 3, CV_64F);
    int frame_id_ = 0;
    
    cv::Mat get_homography_optimized(const cv::Mat& g1, const cv::Mat& g2) {
        // 使用优化后的特征点检测
        std::vector<cv::Point2f> pts1 = cuda_goodFeaturesToTrack_optimized(
            g1, memory_pool_.get(), params_.MAX_FEATURES, 
            params_.FEATURE_QUALITY, params_.MIN_DISTANCE);
        
        if (pts1.empty()) return cv::Mat::eye(3, 3, CV_64F);
        
        // 预分配vectors
        static std::vector<uchar> status;
        static std::vector<float> err;
        static std::vector<cv::Point2f> pts2;
        
        if (status.size() != pts1.size()) {
            status.resize(pts1.size());
            err.resize(pts1.size());
            pts2.resize(pts1.size());
        }
        
        cv::calcOpticalFlowPyrLK(g1, g2, pts1, pts2, status, err);
        
        // 筛选有效点 - 避免频繁的vector操作
        static std::vector<cv::Point2f> p1, p2;
        p1.clear(); p2.clear();
        p1.reserve(pts1.size()); p2.reserve(pts1.size());
        
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                p1.push_back(pts1[i]);
                p2.push_back(pts2[i]);
            }
        }
        
        if (static_cast<int>(p1.size()) < params_.MIN_TRACK_POINTS) {
            return cv::Mat::eye(3, 3, CV_64F);
        }
        
        cv::Mat mask;
        cv::Mat H = cv::findHomography(p2, p1, cv::RANSAC, 
                                      params_.RANSAC_THRESHOLD, mask);
        
        if (H.empty() || cv::countNonZero(mask) < params_.MIN_INLIERS) {
            return cv::Mat::eye(3, 3, CV_64F);
        }
        
        return H;
    }
};

// 工厂函数
VehicleParkingDetect* createVehicleParkingDetectOptimized() {
    return new VehicleParkingDetectOptimized();
}

void destroyVehicleParkingDetectOptimized(VehicleParkingDetect* detector) {
    delete detector;
}
