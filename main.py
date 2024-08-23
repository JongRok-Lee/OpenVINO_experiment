from pathlib import Path
from ultralytics import YOLOv10
from notebook_utils import download_file, VideoPlayer

import os, collections
import cv2
import time
import numpy as np
import openvino as ov
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--show", type=bool, default=False, help="Visualization")
    parser.add_argument("--device", type=str, help="Device to run the model")
    return parser.parse_args()

# Main processing function to run object detection.
def run_object_detection(
    det_model,
    source=0,
    flip=False,
    show=False,
    skip_first_frames=0,
):
    player = None
    elapsed_time = 0
    cnt = 0
    try:
        # Create a video player to play with target fps.
        player = VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start capturing.
        player.start()
        if show:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            # Get the results.
            input_image = np.array(frame)

            start_time = time.time()
            detections = det_model(input_image, iou=0.45, conf=0.2, verbose=False)
            stop_time = time.time()
            frame = detections[0].plot()

            processing_times.append(stop_time - start_time)
            elapsed_time += processing_times[-1]
            cnt += 1
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            if show:
                cv2.putText(
                    img=frame,
                    text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    org=(20, 40),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=f_width / 1000,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                # Use this workaround if there is flickering.
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if show:
            cv2.destroyAllWindows()
    elapsed_time = elapsed_time / cnt * 1000
    print(f"Elapsed time: {elapsed_time:.1f}ms ({1000/elapsed_time:.1f} FPS)")


if __name__ == '__main__':
    args = parse_args()
    models_dir, model_name = Path("./models"), "yolov10n"
    IMAGE_PATH = Path("./data/coco_bike.jpg")
    VIDEO_SOURCE = "traffic.mp4"

    ov_fp16_model_path =  models_dir / f"FP16_openvino_model/{model_name}.xml"
    ov_fp32_model_path = models_dir / f"FP32_openvino_model/{model_name}.xml"
    ov_int8_model_path = models_dir / "INT8_openvino_model" / f"{model_name}.xml"
    
    ov_fp16_model_weights = ov_fp16_model_path.with_suffix(".bin")
    ov_fp32_model_weights = ov_fp32_model_path.with_suffix(".bin")
    ov_int8_model_weights = ov_int8_model_path.with_suffix(".bin")
    print(f"Size of FP32 model is {ov_fp32_model_weights.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Size of FP16 model is {ov_fp16_model_weights.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Size of int8 model is {ov_int8_model_weights.stat().st_size / 1024 / 1024:.2f} MB")

    core = ov.Core()
    device = device = args.device if args.device else "CPU"

    ov_fp16_model = core.read_model(ov_fp16_model_path)
    ov_fp32_model = core.read_model(ov_fp32_model_path)
    ov_int8_model = core.read_model(ov_int8_model_path)

    if "GPU" == device:
        ov_fp16_model.reshape({0: [1, 3, 640, 640]})
        ov_fp32_model.reshape({0: [1, 3, 640, 640]})
        ov_int8_model.reshape({0: [1, 3, 640, 640]})
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        ov_fp32_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES",
                        "CAPABILITY": "FP32",}
    else:
        ov_config = {}
        ov_fp32_config = {}

    det_fp16_compiled = core.compile_model(ov_fp16_model, device, config=ov_config)
    det_fp32_compiled = core.compile_model(ov_fp32_model, device, config=ov_fp32_config)
    det_int8_compiled = core.compile_model(ov_int8_model, device, config=ov_config)

    ov_yolo_fp16_model = YOLOv10(ov_fp16_model_path.parent, task="detect")
    _ = ov_yolo_fp16_model(IMAGE_PATH, iou=0.45, conf=0.2)
    ov_yolo_fp16_model.predictor.model.ov_compiled_model = det_fp16_compiled

    ov_yolo_fp32_model = YOLOv10(ov_fp16_model_path.parent, task="detect")
    _ = ov_yolo_fp32_model(IMAGE_PATH, iou=0.45, conf=0.2)
    ov_yolo_fp32_model.predictor.model.ov_compiled_model = det_fp32_compiled

    ov_yolo_int8_model = YOLOv10(ov_int8_model_path.parent, task="detect")
    _ = ov_yolo_int8_model(IMAGE_PATH, iou=0.45, conf=0.2)
    ov_yolo_int8_model.predictor.model.ov_compiled_model = det_int8_compiled

    run_object_detection(ov_yolo_fp32_model, VIDEO_SOURCE, False, args.show)
    run_object_detection(ov_yolo_fp16_model, VIDEO_SOURCE, False, args.show)
    run_object_detection(ov_yolo_int8_model, VIDEO_SOURCE, False, args.show)
