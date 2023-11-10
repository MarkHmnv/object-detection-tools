import argparse
import os
import cv2

from pathlib import Path
from ultralytics import YOLO


def draw_boxes(image, model, conf=0.4, half=True):
    """
    Draw bounding boxes and labels on an image using a given model.

    :param image: The input image to draw boxes on.
    :param model: YOLOv8 model used for object detection.
    :param conf: The confidence threshold for the bounding boxes.
    :param half: Whether to use half precision for the model.

    :return: The image with bounding boxes and labels drawn.
    """
    results = model(image, conf, half)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1)
    return image


def label_video(video_path, output_directory, model_path, conf=0.4, half=True):
    """
    Label the video in the given path using the provided model and save the results in the output directory.

    :param video_path: The path of the video to be labeled.
    :param output_directory: The directory path where the labeled results will be saved.
    :param model_path: YOLOv8 model path used for labeling the video.
    :param conf: The confidence threshold for the bounding boxes.
    :param half: Whether to use half precision for the model.
    :return: None
    """
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_directory / (os.path.basename(video_path) + ".mp4"), fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame = draw_boxes(frame, model, conf, half)

        out.write(frame)

    video.release()
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="\"Label the video in the given directory using the provided model\"")
    parser.add_argument("-v", "--video_path",
                        help="Path of the video to be labeled.", required=True)
    parser.add_argument("-o", "--output_directory",
                        help="The directory path where the labeled video will be saved.", required=True)
    parser.add_argument("-m", "--model",
                        help="YOLOv8 model path used for labeling the video.", required=True)
    parser.add_argument("-c", "--conf",
                        help="The confidence threshold for the bounding boxes.", type=float, default=0.4)
    parser.add_argument("-h", "--half",
                        help="Whether to use half precision for the model.", action='store_true')

    args = parser.parse_args()

    image_directory = Path(args.image_directory)
    output_directory = Path(args.output_directory)
    model = Path(args.model)
    conf = args.conf
    half = args.half

    label_video(image_directory, output_directory, model, conf, half)
