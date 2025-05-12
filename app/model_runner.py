import numpy as np
import cv2
import torch
import pathlib
from l2cs import select_device, Pipeline
from collections import defaultdict

CWD = pathlib.Path.cwd()

def classify_gaze_direction(pitch, yaw, pitch_thresh=0.15, yaw_thresh=0.15):
    """
    将 pitch/yaw（弧度）分类为 ['up', 'down', 'left', 'right', 'center']。
    - pitch_thresh: 俯仰角分类阈值，推荐 0.15 rad ≈ 8.6 度
    - yaw_thresh: 偏航角分类阈值，推荐 0.15 rad ≈ 8.6 度

    错误的。这个模型里，yaw代表俯仰角（yaw大于0为上），pitch代表左右（pitch>0为右）。
    """
    # 正视判断
    if abs(pitch) <= pitch_thresh and abs(yaw) <= yaw_thresh:
        return "center"
    # 上下优先
    if abs(yaw) > abs(pitch):
        return "up" if yaw> 0 else "down"
    # 左右
    else:
        return "right" if pitch > 0 else "left"

class GazeModel:
    def __init__(self, device='cpu', arch='ResNet50'):
        self.pipeline = Pipeline(
            weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
            arch=arch,
            device=select_device(device, batch_size=1)
        )

    def analyze_frame(self, frame):
        results = self.pipeline.step(frame)
        if results.pitch.shape[0] > 0:
            direction = classify_gaze_direction(results.pitch[0], results.yaw[0])
            pitch_deg = results.pitch[0] * 180.0 / np.pi
            yaw_deg = results.yaw[0] * 180.0 / np.pi
        else:
            direction = "Not detected"
            pitch_deg, yaw_deg = 0.0, 0.0
        return {
            "direction": direction,
            "pitch": float(pitch_deg),
            "yaw": float(yaw_deg)
        }

    def analyze_video(self, video_file, fps=30.0, resize=None):
        directions = []  # (timestamp, direction)
        cap = cv2.VideoCapture(video_file)
        total_frames = 0

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        sampling_interval = int(original_fps // fps) if original_fps >= fps else 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            frame_idx += 1
            if frame_idx % sampling_interval != 0:
                continue

            if resize:
                frame = cv2.resize(frame, resize)
            else:
                resize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            timestamp = frame_idx / original_fps
            result = self.analyze_frame(frame)
            directions.append((timestamp, result["direction"]))

        cap.release()

        # 统计 gaze 时长
        stat = defaultdict(float)
        for _, d in directions:
            stat[d] += 1.0 / fps  # 每帧占 1/fps 秒

        # 生成方向对应的时间段
        segments = defaultdict(list)
        if directions:
            start_time = directions[0][0]
            current_dir = directions[0][1]

            for i in range(1, len(directions)):
                t, d = directions[i]
                if d != current_dir:
                    end_time = t
                    segments[current_dir].append([start_time, end_time])
                    current_dir = d
                    start_time = t

            last_end_time = total_frames / original_fps
            segments[current_dir].append([start_time, last_end_time])

        return {
            "total_duration": directions[-1][0] if directions else 0,
            "fps": fps,
            "resolution": list(resize),
            "direction_stat_seconds": dict(stat),
            "direction_segments": dict(segments)
        }