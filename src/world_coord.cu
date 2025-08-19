#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <vector>

// 每个线程处理一个像素
__global__ void kernel_world_coord(
    const float* __restrict__ x_in,
    const float* __restrict__ y_in,
    float* __restrict__ x_out,
    float* __restrict__ y_out,
    const double* H,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = x_in[i];
    float py = y_in[i];

    double w  = H[6] * px + H[7] * py + H[8];
    double qx = (H[0] * px + H[1] * py + H[2]) / w;
    double qy = (H[3] * px + H[4] * py + H[5]) / w;

    x_out[i] = static_cast<float>(qx);
    y_out[i] = static_cast<float>(qy);
}

// 主机端接口：输入 pts，输出 world
void launch_world_coord(const std::vector<cv::Point2f>& pts,
                               const cv::Mat& H,
                               std::vector<cv::Point2f>& world)
{
    int N = static_cast<int>(pts.size());
    world.resize(N);

    cv::Mat Hd; H.convertTo(Hd, CV_64F);
    double h[9];
    for (int i = 0; i < 9; ++i) h[i] = Hd.ptr<double>(0)[i];

    float  *d_x, *d_y, *d_ox, *d_oy;
    double *d_H;
    cudaMalloc(&d_x,  N * sizeof(float));
    cudaMalloc(&d_y,  N * sizeof(float));
    cudaMalloc(&d_ox, N * sizeof(float));
    cudaMalloc(&d_oy, N * sizeof(float));
    cudaMalloc(&d_H,  9 * sizeof(double));

    std::vector<float> xs(N), ys(N);
    for (int i = 0; i < N; ++i) { xs[i] = pts[i].x; ys[i] = pts[i].y; }
    cudaMemcpy(d_x, xs.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, ys.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, h, 9 * sizeof(double), cudaMemcpyHostToDevice);

    constexpr int block = 256;
    int grid = (N + block - 1) / block;
    kernel_world_coord<<<grid, block>>>(d_x, d_y, d_ox, d_oy, d_H, N);

    std::vector<float> ox(N), oy(N);
    cudaMemcpy(ox.data(), d_ox, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(oy.data(), d_oy, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) world[i] = cv::Point2f(ox[i], oy[i]);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ox); cudaFree(d_oy); cudaFree(d_H);
}