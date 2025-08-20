#include <opencv2/opencv.hpp>
#include <deque>
#include <unordered_map>
#include <chrono>
#include "byte_track.h"
#include "detect.h"
#include "vehicle_parking_detect.h"
using namespace std;

// 声明优化版本的工厂函数
VehicleParkingDetect* createVehicleParkingDetectOptimized();
void destroyVehicleParkingDetectOptimized(VehicleParkingDetect* detector);

void launch_world_coord(const std::vector<cv::Point2f>& pts,
                        const cv::Mat& H,
                        std::vector<cv::Point2f>& world);
// cv::Mat get_homography_full_cuda(const cv::Mat& g1, const cv::Mat& g2); // 完全CUDA版本      
std::vector<cv::Point2f> cuda_goodFeaturesToTrack(const cv::Mat& image, 
                                                  int max_corners = 800,
                                                  double quality_level = 0.02,
                                                  double min_distance = 10.0);       
// ================= 用户可调参数 =================
const std::string VIDEO_PATH = "/home/ubuntu/Desktop/DJI_20250704170646_0001_V_cut.mp4";
const std::set<int> VEHICLE_IDS = {0}; // COCO: car=2, bus=5, truck=7



int main() {

    
    xtkj::IDetect*     detector = xtkj::createDetect();
    xtkj::ITracker*    tracker  = xtkj::createTracker(30, 30, 0.5, 0.6, 0.8);
    tracker->init(30, 30, 0.5, 0.6, 0.8);

    VehicleParkingDetect* vehicleParkingDetect = createVehicleParkingDetectOptimized();
    
    // 初始化车辆停车检测参数
    VehicleParkingInitParams parkingParams;
    parkingParams.K = 4;
    parkingParams.EPS_WORLD = 1.0;
    parkingParams.MIN_SPEED_FRAMES = 3;
    parkingParams.RESET_EVERY = 200;
    parkingParams.MAX_FEATURES = 800;
    parkingParams.FEATURE_QUALITY = 0.02;
    parkingParams.MIN_DISTANCE = 10;
    parkingParams.MIN_TRACK_POINTS = 80;
    parkingParams.RANSAC_THRESHOLD = 3.0;
    parkingParams.MIN_INLIERS = 80;
    
    if (!vehicleParkingDetect->init(parkingParams)) {
        std::cerr << "Failed to initialize vehicle parking detector" << std::endl;
        return -1;
    }
    
    AlgorConfig        algorConfig;
    algorConfig.algorName_ = "object_detect";
    algorConfig.model_path = "best.onnx";
    int batch                  = 1;
    detect_result_group_t**   outs        = new detect_result_group_t*[batch];
    int                       frame_count = 0;

    
    algorConfig.img_size       = 640;
    algorConfig.conf_thresh    = 0.25f;
    algorConfig.iou_thresh     = 0.2f;
    algorConfig.max_batch_size = batch;
    algorConfig.min_opt        = 1;
    algorConfig.mid_opt        = 16;
    algorConfig.max_opt        = 32;
    algorConfig.is_ultralytics = 1;  // 是否使用ultralytics模型
    algorConfig.gpu_id         = 0;
    detector->init(algorConfig);
    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) return -1;


    int frame_id = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        cv::Mat gray;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        int max_dim = std::max(gray.rows, gray.cols);
        double parking_scale = 640.0 / max_dim;
        cv::Size parking_size(static_cast<int>(gray.cols * parking_scale),
                             static_cast<int>(gray.rows * parking_scale));
        cv::resize(gray, gray, parking_size);
        cv::Mat park_show;
        cv::resize(frame, park_show, parking_size);

       
        vector<long long>       mat_info = {(long long)frame.data,
                                      (long long)frame.cols,
                                      (long long)frame.rows};
        std::vector<std::vector<long long>> mats;
                            
        mats.push_back(mat_info);
        outs[0] = new detect_result_group_t();

        // 执行检测
        detector->forward(mats, outs);
        detect_result_group_t* objects = outs[0];
        int ret = tracker->track(objects, frame.cols, frame.rows);
        std::vector<TrackBox> tracks;
        for (int i = 0; i < objects->count; i++) {
            int x1       = objects->results[i].box.left * parking_scale;
            int y1       = objects->results[i].box.top * parking_scale;
            int x2       = objects->results[i].box.right * parking_scale;
            int y2       = objects->results[i].box.bottom * parking_scale;
            int id = objects->results[i].track_id;
            int cls_id   = objects->results[i].cls_id;
            tracks.push_back({id, cv::Rect(x1, y1, x2 - x1, y2 - y1), cls_id,
                              objects->results[i].prop, false, 0.0});
        
        }
        auto start_time = std::chrono::high_resolution_clock::now();
        ret = vehicleParkingDetect->detect(gray, tracks);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Vehicle parking detection took " << duration.count() << " ms" << std::endl;
        for(auto track:tracks){
            int x1       = track.box.x;
            int y1       = track.box.y;
            int x2       = x1 + track.box.width;
            int y2       = y1 + track.box.height;
            cv::Scalar color = track.is_still ? cv::Scalar(0, 0, 255)
                                           : cv::Scalar(0, 255, 0);
            cv::rectangle(park_show, cv::Point(x1,y1), cv::Point(x2,y2), color, 2);
            std::string txt = cv::format("%d delta:%.2f", track.track_id, track.delta);
            if (track.is_still) txt += " PARKED";
            cv::putText(park_show, txt, cv::Point(x1, y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
        cv::imshow("World-still detect", park_show);
        if (cv::waitKey(1) == 27) break;
        ++frame_id;
    }
    
    // 清理资源
    destroyVehicleParkingDetectOptimized(vehicleParkingDetect);
    delete tracker;
    delete detector;
    delete[] outs;
    
    return 0;
}