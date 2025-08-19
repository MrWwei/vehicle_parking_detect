import cv2
import numpy as np
from ultralytics import YOLO
# from ultralytics.tracker import register_tracker

# -------------------- 参数 --------------------
VIDEO_PATH = '/home/ubuntu/Desktop/DJI_20250704170646_0001_V_cut.mp4'
CLS_VEHICLE = [0]          # COCO car, motor, bus, truck
STILL_FRAMES = 30                   # 连续多少帧不动算静止
STILL_PIXEL = 3.0                   # 世界坐标下允许的像素漂移
# ---------------------------------------------

model = YOLO('car_yolov5nu.pt')
# register_tracker(model, 'bytetrack.yaml')   # 用官方bytetrack
print('模型加载完成！')
cap = cv2.VideoCapture(VIDEO_PATH)
prev_gray = None
prev_pts = None
# id -> 队列（存放世界坐标）
world_hist = {}

def get_homography(gray1, gray2):
    pts1 = cv2.goodFeaturesToTrack(gray1, 500, 0.01, 7)
    if pts1 is None:
        return None
    pts2, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None)
    st = st.reshape(-1).astype(bool)
    pts1, pts2 = pts1[st], pts2[st]
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return H if mask.sum() > 30 else None

while True:
    ok, frame = cap.read()
    print('读取视频帧...')
    if not ok:
        print('视频读取完毕！ 退出...')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H = get_homography(prev_gray, gray) if prev_gray is not None else None
    prev_gray = gray
    
    print('检测目标...')

    # ① YOLO + ByteTrack
    results = model.track(frame, persist=True, classes=CLS_VEHICLE, verbose=False)[0]
    boxes = results.boxes
    if boxes.id is None:            # 没检测到目标
        print('未检测到目标，继续下一帧...')
        continue
    ids = boxes.id.cpu().int().tolist()
    xyxys = boxes.xyxy.cpu().numpy()

    for idx, (xyxy, id_) in enumerate(zip(xyxys, ids)):
        cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
        pt = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        # ② 补偿相机运动 → 得到世界坐标
        if H is not None:
            world_pt = cv2.perspectiveTransform(pt, H)[0][0]
        else:
            world_pt = pt[0][0]     # 首帧无补偿
        world_hist.setdefault(id_, []).append(world_pt)

        # ③ 判静止
        hist = world_hist[id_]
        if len(hist) > STILL_FRAMES:
            hist = hist[-STILL_FRAMES:]
            dists = np.linalg.norm(np.diff(np.array(hist), axis=0), axis=1)
            still = np.sum(dists) < STILL_PIXEL
            if still:
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                cv2.putText(frame, f'{id_} STILL',
                            (int(xyxy[0]), int(xyxy[1])-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('YOLOv8+ByteTrack+静止检测', frame)
    print('显示视频帧...')
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()