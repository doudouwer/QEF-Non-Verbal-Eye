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
from evaluate import classify_gaze_direction

CWD = pathlib.Path.cwd()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)
    parser.add_argument(
        '--video', dest='video_path', help='Path to video file (optional, overrides --cam)',
        default=None, type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
    if args.video_path is not None:
        cap = cv2.VideoCapture(args.video_path)
        print(f"[INFO] Using video file: {args.video_path}")
    else:
        cap = cv2.VideoCapture(args.cam_id)
        print(f"[INFO] Using webcam with ID {args.cam_id}")

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:

            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            # Process frame
            results = gaze_pipeline.step(frame)

            # Visualize output
            frame = render(frame, results)
           
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            if results.pitch.shape[0] > 0:
                direction = classify_gaze_direction(results.pitch[0], results.yaw[0])
                pitch_deg = results.pitch[0] * 180.0 / np.pi
                yaw_deg = results.yaw[0] * 180.0 / np.pi
            else:
                direction = "Not detected"
                pitch_deg = 0.0
                yaw_deg = 0.0
            cv2.putText(frame, f'Direction: {direction}', (10, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)

            cv2.putText(frame, f'Pitch: {pitch_deg:.1f}°', (10, 70),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Yaw: {yaw_deg:.1f}°', (10, 95),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()