import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
# from ultralytics.tracker import register_tracker

# ================= 用户可调参数 =================
VIDEO_PATH   = 'F:/data/gaokong/Drone/DJI_20251128212857_0003_S/test_cut.mp4'      # 输入视频
# VIDEO_PATH   = 'F:/data/gaokong/Drone/chizhou/chizhou_cut0.mp4'      # 输入视频
OUTPUT_PATH  = 'static_vehicle_detection_output.mp4'  # 输出视频路径
VEHICLE_IDS  = {0}     # COCO: car, motor, bus, truck

# 静止判定参数
K            = 4               # 连续帧数：增加到5帧提高稳定性
EPS_WORLD    = 2.0             # 世界坐标位移阈值：降低阈值更严格
MIN_SPEED_FRAMES = 3           # 最少静止帧数：需要连续静止帧
RESET_EVERY  = 200             # 参考帧重置频率：更频繁重置避免累积误差

# 特征点检测参数
MAX_FEATURES = 800             # 增加特征点数量
FEATURE_QUALITY = 0.02         # 提高特征点质量阈值
MIN_DISTANCE = 10              # 增加特征点间最小距离
MIN_TRACK_POINTS = 80          # 增加最少跟踪点数

# 单应性计算参数
RANSAC_THRESHOLD = 3.0         # 降低RANSAC阈值提高精度
MIN_INLIERS = 80               # 增加最少内点数
# ===============================================

model = YOLO('pingyang_car_u.pt')
# register_tracker(model, 'bytetrack.yaml')

cap = cv2.VideoCapture(VIDEO_PATH)

# 获取视频属性
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

prev_gray = None
ref_H_acc   = np.eye(3)         # 累积单应：当前帧 → 参考帧
frame_id    = 0

# id -> deque(maxlen=K) 保存世界坐标
world_hist = defaultdict(lambda: deque(maxlen=K))
# 记录每个车辆的静止状态
still_count = defaultdict(int)  # 连续静止帧计数

# ---------- 工具 ----------
def get_homography(g1, g2):
    """计算两帧图像间的单应性矩阵，使用改进的参数"""
    pts1 = cv2.goodFeaturesToTrack(g1, MAX_FEATURES, FEATURE_QUALITY, MIN_DISTANCE)
    if pts1 is None:
        return np.eye(3)
    
    pts2, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts1, None)
    st = st.ravel().astype(bool)
    
    if st.sum() < MIN_TRACK_POINTS:
        return np.eye(3)
    
    pts1, pts2 = pts1[st], pts2[st]
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_THRESHOLD)
    
    if H is None or mask.sum() < MIN_INLIERS:
        return np.eye(3)
    
    return H

def world_coord(H, x, y):
    """把像素坐标映射到世界坐标（齐次归一）"""
    p = np.array([[x, y, 1]], dtype=np.float64).T
    q = H @ p
    q /= q[2]
    return q[:2, 0]

# ---------- 主循环 ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ① 计算当前帧 → 上一帧的单应 H_curr
    H_curr = get_homography(prev_gray, gray) if prev_gray is not None else np.eye(3)
    prev_gray = gray

    # ② 累积单应：当前帧 → 参考帧
    ref_H_acc = H_curr @ ref_H_acc

    # ③ 每 RESET_EVERY 帧重置参考帧
    if frame_id % RESET_EVERY == 0:
        ref_H_acc = np.eye(3)
        world_hist.clear()
        still_count.clear()  # 清空静止计数

    # ④ YOLOv8 + ByteTrack
    results = model.track(frame, persist=True, classes=list(VEHICLE_IDS), verbose=False)[0]
    boxes = results.boxes
    if boxes.id is None:
        frame_id += 1
        # 即使没有检测到车辆，也要写入原始帧
        out.write(frame)
        cv2.imshow('World-still detect', frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    ids = boxes.id.cpu().int().tolist()
    xyxys = boxes.xyxy.cpu().numpy().astype(int)

    # ⑤ 计算世界坐标 & 静止判决
    for xyxy, id_ in zip(xyxys, ids):
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        wp = world_coord(ref_H_acc, cx, cy)
        world_hist[id_].append(wp)

        hist = world_hist[id_]
        is_currently_still = False
        delta = 0.0
        
        # 计算位移
        if len(hist) >= 2:
            # 计算最近几帧的平均位移
            deltas = []
            for i in range(1, min(len(hist), K)):
                deltas.append(np.linalg.norm(hist[-1] - hist[-i-1]))
            
            if deltas:
                delta = np.mean(deltas)  # 使用平均位移
                is_currently_still = delta < EPS_WORLD
        
        # 更新静止计数
        if is_currently_still:
            still_count[id_] += 1
        else:
            still_count[id_] = 0
        
        # 最终判定：需要连续静止足够帧数
        final_still = still_count[id_] >= MIN_SPEED_FRAMES
        
        # 绘制结果
        color = (0, 0, 255) if final_still else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 显示信息：ID, 位移, 静止帧数
        text = f'{id_} {delta:.1f}px {still_count[id_]}f'
        if final_still:
            text += ' PARKED'
        
        cv2.putText(frame, text,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 将处理后的帧写入输出视频
    out.write(frame)
    
    cv2.imshow('World-still detect', frame)
    frame_id += 1
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()  # 关闭视频写入器
cv2.destroyAllWindows()
print(f"输出视频已保存至: {OUTPUT_PATH}")
