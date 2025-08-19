#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <vector>

// CPU版本的mean optical flow计算
float mean_optical_flow(const cv::Mat&  prev_gray,
                        const cv::Mat&  curr_gray,
                        const cv::Rect& bbox)
{
    cv::Mat roi_prev = prev_gray(cv::Range(bbox.y, bbox.y + bbox.height),
                                 cv::Range(bbox.x, bbox.x + bbox.width));
    cv::Mat roi_curr = curr_gray(cv::Range(bbox.y, bbox.y + bbox.height),
                                 cv::Range(bbox.x, bbox.x + bbox.width));

    if (roi_prev.empty() || roi_curr.empty()) {
        return 0.0f;
    }

    std::vector<cv::Point2f> p0;
    cv::goodFeaturesToTrack(roi_prev, p0, 30, 0.01, 3);

    if (p0.empty()) {
        return 0.0f;
    }

    std::vector<cv::Point2f> p1;
    std::vector<uchar>       st;
    std::vector<float>       err;
    cv::calcOpticalFlowPyrLK(
        roi_prev, roi_curr, p0, p1, st, err, cv::Size(15, 15), 2,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10,
                         0.03));

    float sum_disp = 0.0f;
    int   count    = 0;
    for (size_t i = 0; i < p0.size(); ++i) {
        if (st[i]) {
            float dx = p1[i].x - p0[i].x;
            float dy = p1[i].y - p0[i].y;
            sum_disp += std::sqrt(dx * dx + dy * dy);
            count++;
        }
    }

    return count > 0 ? sum_disp / count : 0.0f;
}

// GPU版本的mean optical flow计算 - 使用与CPU版本相同的稀疏光流算法
float mean_optical_flow_gpu(const cv::cuda::GpuMat& prev_gray,
                           const cv::cuda::GpuMat& curr_gray,
                           const cv::Rect& bbox,
                           cv::cuda::Stream& stream = cv::cuda::Stream::Null())
{
    // 提取ROI区域
    cv::cuda::GpuMat roi_prev = prev_gray(bbox);
    cv::cuda::GpuMat roi_curr = curr_gray(bbox);
    
    if (roi_prev.empty() || roi_curr.empty()) {
        return 0.0f;
    }
    
    // 下载到CPU进行特征点检测和稀疏光流计算
    // 因为OpenCV的CUDA版本对这些算法支持有限，使用CPU版本确保结果一致
    cv::Mat roi_prev_cpu, roi_curr_cpu;
    roi_prev.download(roi_prev_cpu);
    roi_curr.download(roi_curr_cpu);
    
    // 使用与CPU版本完全相同的算法
    std::vector<cv::Point2f> p0;
    cv::goodFeaturesToTrack(roi_prev_cpu, p0, 30, 0.01, 3);

    if (p0.empty()) {
        return 0.0f;
    }

    std::vector<cv::Point2f> p1;
    std::vector<uchar>       st;
    std::vector<float>       err;
    cv::calcOpticalFlowPyrLK(
        roi_prev_cpu, roi_curr_cpu, p0, p1, st, err, cv::Size(15, 15), 2,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10,
                         0.03));

    float sum_disp = 0.0f;
    int   count    = 0;
    for (size_t i = 0; i < p0.size(); ++i) {
        if (st[i]) {
            float dx = p1[i].x - p0[i].x;
            float dy = p1[i].y - p0[i].y;
            sum_disp += std::sqrt(dx * dx + dy * dy);
            count++;
        }
    }

    return count > 0 ? sum_disp / count : 0.0f;
}