#include "vehicle_parking_detect.h"


void launch_world_coord(const std::vector<cv::Point2f>& pts,
                        const cv::Mat& H,
                        std::vector<cv::Point2f>& world); 
std::vector<cv::Point2f> cuda_goodFeaturesToTrack(const cv::Mat& image, 
                                                  int max_corners = 800,
                                                  double quality_level = 0.02,
                                                  double min_distance = 10.0);   


class VehicleParkingDetectImpl : public VehicleParkingDetect {
public:
    VehicleParkingDetectImpl() {
        // 初始化检测器
        
    }
    ~VehicleParkingDetectImpl() override {
        // 清理资源
    }
    bool init(VehicleParkingInitParams& params) override {
        params_ = params;  // 保存初始化参数
        // 初始化检测器逻辑
        // 例如加载模型、设置参数等
        return true;  // 返回true表示初始化成功
    }
    bool detect(const cv::Mat& gray_mat,  std::vector<TrackBox>& tracks) override {
        // 执行检测逻辑
        bool final_still = false;
        // auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat gray = gray_mat.clone();
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "clone took " << duration.count() << " ms" << std::endl;
        if(gray.channels() != 1) {
            cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        }
        // auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat H_curr = prev_gray_.empty() ? cv::Mat::eye(3, 3, CV_64F)
                                           : get_homography(prev_gray_, gray);
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "get_homography took " << duration.count() << " ms" << std::endl;
        prev_gray_ = gray.clone();
        
        
        
        ref_H_acc_ = H_curr * ref_H_acc_;
         // ③ 重置参考帧
        if (frame_id_ % params_.RESET_EVERY == 0) {
            ref_H_acc_ = cv::Mat::eye(3, 3, CV_64F);
            world_hist_.clear();
            still_count_.clear();
        }
         // 准备坐标
        std::vector<cv::Point2f> pts;
        for(auto track_box: tracks) {
            int center_x = track_box.box.x + track_box.box.width / 2;
            int center_y = track_box.box.y + track_box.box.height / 2;
            pts.emplace_back(center_x, center_y);
        }
        // CUDA 世界坐标
        std::vector<cv::Point2f> world;
        if (!pts.empty()){
            launch_world_coord(pts, ref_H_acc_, world);
        }
        int world_idx = 0;
        for(auto &track_box: tracks) {
            int id = track_box.track_id;
        
            world_hist_[id].push_back(world[world_idx++]);
            if (int(world_hist_[id].size()) > params_.K) world_hist_[id].pop_front();

            bool is_still = false;
            double delta = 0;
            if (world_hist_[id].size() >= 2) {
                double sum = 0;
                for (size_t i = 1; i < world_hist_[id].size(); ++i)
                    sum += cv::norm(world_hist_[id][i] - world_hist_[id][0]);
                delta = sum / (world_hist_[id].size() - 1);
                is_still = delta < params_.EPS_WORLD;
                track_box.delta = delta;
            }

            still_count_[id] = is_still ? still_count_[id] + 1 : 0;
            final_still = still_count_[id] >= params_.MIN_SPEED_FRAMES;
            track_box.is_still = final_still;
        }
        
        frame_id_++;
        return true;  // 返回true表示检测成功
    }
private:
    VehicleParkingInitParams params_;  // 存储初始化参数
    std::unordered_map<int, std::deque<cv::Point2f>> world_hist_;
    std::unordered_map<int, int> still_count_;
    cv::Mat prev_gray_;
    cv::Mat ref_H_acc_ = cv::Mat::eye(3, 3, CV_64F);
    int frame_id_ = 0;  // 帧计数器
    // const int K = 4;  // 世界坐标历史长度
    // const double EPS_WORLD = 2.0;  // 世界坐标容差
    // const int MIN_SPEED_FRAMES = 3;  // 最小静止帧
    // const int RESET_EVERY = 200;  // 重置间隔

private:
    cv::Mat get_homography(const cv::Mat& g1, const cv::Mat& g2) {
        std::vector<cv::Point2f> pts1, pts2;
        pts1 = cuda_goodFeaturesToTrack(g1);
        
        if (pts1.empty()) return cv::Mat::eye(3, 3, CV_64F);

        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(g1, g2, pts1, pts2, status, err);

        std::vector<cv::Point2f> p1, p2;
        for (size_t i = 0; i < status.size(); ++i)
            if (status[i]) { p1.push_back(pts1[i]); p2.push_back(pts2[i]); }
        if (int(p1.size()) < params_.MIN_TRACK_POINTS) return cv::Mat::eye(3, 3, CV_64F);

        cv::Mat mask;
        cv::Mat H = cv::findHomography(p2, p1, cv::RANSAC, params_.RANSAC_THRESHOLD, mask);
        if (H.empty() || cv::countNonZero(mask) < params_.MIN_INLIERS)
            return cv::Mat::eye(3, 3, CV_64F);
        return H;
    }
};
VehicleParkingDetect* createVehicleParkingDetect() {
    return new VehicleParkingDetectImpl();
}
void destroyVehicleParkingDetect(VehicleParkingDetect* detector) {
    delete detector;
}