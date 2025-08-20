
#ifndef VEHICLE_PARKING_DETECT_H
#define VEHICLE_PARKING_DETECT_H
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

struct VehicleParkingInitParams{
    int K = 4;
    double EPS_WORLD = 2.0;
    int MIN_SPEED_FRAMES = 3;
    int RESET_EVERY = 200;

    int MAX_FEATURES = 800;
    double FEATURE_QUALITY = 0.02;
    int MIN_DISTANCE = 10;
    int MIN_TRACK_POINTS = 20;
    double RANSAC_THRESHOLD = 3.0;
    int MIN_INLIERS = 80;
};

struct TrackBox {
    int track_id;
    cv::Rect box;
    int cls_id;  // 类别ID
    float confidence;  // 置信度
    bool is_still;  // 是否静止
    double delta;  // 平均位移
    // 构造函数
    TrackBox(int id, const cv::Rect& b, int cls, float conf, bool still, double d)
        : track_id(id), box(b), cls_id(cls), confidence(conf), is_still(still), delta(d) {}
    // 默认构造函数
    TrackBox() : track_id(-1), box(cv::Rect()), cls_id(-1), confidence(0.0f), is_still(false), delta(0.0) {}
};
class VehicleParkingDetect {
public:
    virtual ~VehicleParkingDetect() = default;
    // 初始化检测器
    virtual bool init(VehicleParkingInitParams& params) = 0;
    // 执行检测
    virtual bool detect(const cv::Mat& gray_mat, std::vector<TrackBox>& tracks) = 0;
    };

// 工厂函数
VehicleParkingDetect* createVehicleParkingDetectOptimized();
// 销毁检测器
void destroyVehicleParkingDetectOptimized(VehicleParkingDetect* detector);



#endif  // CRUISE_EVENT_H