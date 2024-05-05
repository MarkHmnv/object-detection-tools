import argparse
import os
import cv2

from pathlib import Path
from ultralytics import YOLO


def label_images(image_directory, output_directory, model_path, conf=0.4, half=True):
    """
    Label the images in the given directory using the provided model and save the results in the output directory in txt format.

    :param image_directory: The directory path containing the images to be labeled.
    :param output_directory: The directory path where the labeled results will be saved.
    :param model_path: YOLOv8 model used for labeling the images.
    :param conf: The confidence threshold for the bounding boxes.
    :param half: Whether to use half precision for the model.
    :return: None
    """
    model = YOLO(model_path)
    output_directory.mkdir(exist_ok=True)

    for image_file in os.listdir(image_directory):
        img = cv2.imread(os.path.join(image_directory, image_file))
        filename, _ = os.path.splitext(image_file)
        results = model(img, conf=conf, half=half)

        with open(output_directory / (filename + ".txt"), 'w') as f:
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    x, y, w, h = box.xywhn[0]
                    line = f'{cls} {x} {y} {w} {h}\n'
                    f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="\"Label the images in the given directory using the provided model\"")
    parser.add_argument("-i", "--image_directory",
                        help="The directory path containing the images to be labeled.", required=True)
    parser.add_argument("-o", "--output_directory",
                        help="The directory path where the labeled results will be saved.", required=True)
    parser.add_argument("-m", "--model",
                        help="YOLOv8 model path used for labeling the images.", required=True)
    parser.add_argument("-c", "--conf",
                        help="The confidence threshold for the bounding boxes.", type=float, default=0.4)
    parser.add_argument("--half",
                        help="Whether to use half precision for the model.", action='store_true')

    args = parser.parse_args()

    image_directory = Path(args.image_directory)
    output_directory = Path(args.output_directory)
    model = Path(args.model)
    conf = args.conf
    half = args.half

    label_images(image_directory, output_directory, model, conf=conf, half=half)
