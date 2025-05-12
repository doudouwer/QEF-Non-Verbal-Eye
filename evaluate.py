import argparse
import pathlib
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render

CWD = pathlib.Path.cwd()

import json

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

def record_gaze_segments(gaze_pipeline, cam, output_json_path: str, fps: float = 5.0):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    segments = []
    current_gesture = None
    start_time = 0.0
    frame_idx = 0

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = gaze_pipeline.step(frame)

            # 默认：若无检测框 → Not detected
            if results.pitch.shape[0] == 0:
                gesture = "Not detected"
            else:
                # 仅用第一个人脸做分析
                pitch = results.pitch[0]
                yaw = results.yaw[0]
                gesture = classify_gaze_direction(pitch, yaw)

            cur_time = frame_idx / fps
            if frame_idx % 5 == 0:
                print(f"[INFO] Processed frame {frame_idx} ({cur_time:.2f} sec) — Gesture: {gesture}")

            if gesture != current_gesture:
                if current_gesture is not None:
                    segments.append({
                        "start_time": start_time,
                        "end_time": cur_time,
                        "detected_gesture": current_gesture
                    })
                current_gesture = gesture
                start_time = cur_time

            frame_idx += 1

    # 最后一段
    end_time = frame_idx / fps
    if current_gesture is not None:
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "detected_gesture": current_gesture
        })

    cap.release()

    # 保存为 JSON
    with open(output_json_path, 'w') as f:
        json.dump(segments, f, indent=2)


def extract_labeled_segments(filepath: str, external_id: str, label_name: str):
    with open(filepath, 'r') as f:
        data = json.load(f)

    external_id = "QEF_eye_tracking_testing_video.mp4"

    matched_feature_ids = set()
    for item in data:
        if item.get("data_row", {}).get("external_id") == external_id:
            projects = item.get("projects", {})
            for project in projects.values():
                labels = project.get("labels", [])
                for label_entry in labels:
                    annotations = label_entry.get("annotations", {})
                    frames = annotations.get("frames", {})
                    for frame in frames.values():
                        classifications = frame.get("classifications", [])
                        for classification in classifications:
                            if classification.get("radio_answer").get("name") == label_name:
                                radio_answer = classification.get("radio_answer", {})
                                feature_id = radio_answer.get("feature_id")
                                if feature_id:
                                    matched_feature_ids.add(feature_id)

    for item in data:
        if item.get("data_row", {}).get("external_id") == external_id:
            frame_rate = item.get("media_attributes", {}).get("frame_rate", 30)

            projects = item.get("projects", {})
            for project in projects.values():
                labels = project.get("labels", [])
                for label_entry in labels:
                    annotations = label_entry.get("annotations", {})
                    segments_dict = annotations.get("segments", {})
                    time_segments = []
                    for fid in matched_feature_ids:
                        if fid in segments_dict:
                            frame_segments = segments_dict[fid]
                            for seg in frame_segments:
                                start_frame, end_frame = seg
                                start_sec = (start_frame - 1) / frame_rate
                                end_sec = end_frame / frame_rate
                                time_segments.append((start_sec, end_sec))
                    return time_segments
    return []

def extract_gesture_segments(filepath: str, label_name: str):
    with open(filepath, 'r') as f:
        data = json.load(f)

    result = []
    for entry in data:
        if entry.get("detected_gesture") == label_name:
            result.append((entry["start_time"], entry["end_time"]))
    return result

def calculate_precision_recall_by_time(predicted_segments, true_segments):
    def overlap_duration(seg1, seg2):
        start1, end1 = seg1
        start2, end2 = seg2
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        return max(0.0, inter_end - inter_start)

    total_overlap = 0.0
    total_predicted = sum(end - start for start, end in predicted_segments)
    total_true = sum(end - start for start, end in true_segments)

    for pred in predicted_segments:
        for true in true_segments:
            total_overlap += overlap_duration(pred, true)

    precision = total_overlap / total_predicted if total_predicted > 0 else 0.0
    recall = total_overlap / total_true if total_true > 0 else 0.0
    return precision, recall

def evaluate_pipeine():
    video_names = [
        "yanjia.mp4"
    ]

    output_rows = []
    output_csv_path = "./results/eye_evaluation_summary.csv"
    label_names = ["down", "up", "left", "right"]

    for video_name in video_names:
        base_video_name = video_name.replace("_speechOnly", "").replace("_SpeechOnly", "")
        print(f"base_video_name: ", base_video_name)

        predicted_path = f"./data/yanjia.json"
        ground_truth_path = "./data/labeled_data_eye_tracking.json"

        for label_name in label_names:
            labeled_segments = extract_labeled_segments(ground_truth_path, base_video_name, label_name)
            predicted_segments = extract_gesture_segments(predicted_path, label_name)
            precision, recall = calculate_precision_recall_by_time(predicted_segments, labeled_segments)

            row = {
                "video_name": video_name,
                "label_name": label_name,
                "labeled_segments": json.dumps(labeled_segments),
                "predicted_segments": json.dumps(predicted_segments),
                "precision": round(precision, 3),
                "recall": round(recall, 3)
            }
            output_rows.append(row)

        os.makedirs("./results", exist_ok=True)
        with open(output_csv_path, "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = ["video_name", "label_name", "labeled_segments", "predicted_segments", "precision", "recall"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.',
        default='models/L2CSNet_gaze360.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    cam = args.cam_id

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=select_device(args.device, batch_size=1)
    )

    output_json_path = f"./results/yanjia.json"
    record_gaze_segments(gaze_pipeline, cam, output_json_path, fps=5.0)
