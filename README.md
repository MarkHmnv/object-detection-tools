# Object Detection Tools
This repository provides a collection of utilities for object detection tasks.

## Tools available

1. blender_render_labeling.py - This is a script to be used with Blender to render a scene and produce bounding box annotations. It's useful for generating synthetic datasets for object detection tasks.
2. label_images.py - This script uses YOLOv8 to label images in a given directory and output the labeled images in a txt format.
3. label_video.py - This script uses YOLOv8 to create labeled boxes on frames of the input video and save them as an output video.

## Dependencies
The dependencies required for these scripts are Ultralytics, OpenCV, Bpy and Numpy.
```shell
pip install -r requirements.txt
```

## How to use
Here are examples of how you'd use each tool:

1. blender_render_labeling.py

Before using read the [explanatory example with Google 3D](https://drive.google.com/file/d/17INOsLq8YfU0K9wwGxGZX5kY5g3soNS6/view?usp=sharing)
```shell
python blender_render_labeling.py 
        --project_dir="/path/to/your/project" 
        --frame_start=0 
        --frame_end=100 
        --tilt_angle=45 
        --altitude=200 
        --FOV=60
```

2. label_images.py
```shell
python label_images.py 
        --image_directory="/path/to/your/image_directory"
        --output_directory="/path/to/your/output_directory" 
        --model="/path/to/your/YOLOv8_model" 
        --conf=0.4
        --half=True
```

3. label_video.py
```shell
python label_video.py 
        --video_path="/path/to/your/video_path"
        --output_directory="/path/to/your/output_directory" 
        --model="/path/to/your/YOLOv8_model" 
        --conf=0.4
        --half=True
```
