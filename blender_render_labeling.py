"""
This is a script to be used with Blender to render a scene and produce bounding box annotations.
The script is mainly useful for generating synthetic dataset for object detection tasks, as it not only renders
the scene but also computes and saves bounding box annotations for specified objects in the scene corresponding
to each frame.

The project directory, start and end frames, camera flight tilt angle, altitude, and FOV are the configurable
parameters of the script.

The script generate labels in YOLO format and save images and labels in corresponding directories within
the project directory.

"""


import argparse

import bpy
import numpy as np
import sys
import shutil
import os
import json

default_config = {
    "project_dir": r"Path\to\project",
    "frame_start": 0,
    "frame_end": 15,
    "tilt_angle": 30,
    "altitude": 200,
    "FOV": 60,
}


class Box:
    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return self.min_x * self.dim_x

    @property
    def y(self):
        return self.dim_y - self.max_y * self.dim_y

    @property
    def width(self):
        return (self.max_x - self.min_x) * self.dim_x

    @property
    def height(self):
        return (self.max_y - self.min_y) * self.dim_y

    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % \
            (self.x, self.y, self.width, self.height)

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return 0, 0, 0, 0
        return self.x, self.y, self.width, self.height


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg cam_ob: Camera object.
    :type cam_ob: :class:`bpy.types.Object`
    :arg me_ob: Untransformed Mesh.
    :type me_ob: :class:`bpy.types.Mesh´
    :return: a Box object (call to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """
    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    me = me_ob.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)

    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)
        center_x = clamp((max(lx) + min(lx)) / 2, 0., 1.)
        center_y = clamp((max(ly) + min(ly)) / 2, 0., 1.)

        bad_bbox = False
        if center_x in (0., 1.) and center_y in (0., 1.):
            bad_bbox = True
        if min(lx) <= 0 and max(lx) >= 1:
            bad_bbox = True
        if min(ly) <= 0 and max(ly) >= 1:
            bad_bbox = True

        if bad_bbox:
            min_x = max_x = min_y = max_y = 0.
        else:
            min_x = clamp(min(lx), 0.0, 1.0)
            max_x = clamp(max(lx), 0.0, 1.0)
            min_y = clamp(min(ly), 0.0, 1.0)
            max_y = clamp(max(ly), 0.0, 1.0)

        r = scene.render
        fac = r.resolution_percentage * 0.01
        dim_x = r.resolution_x * fac
        dim_y = r.resolution_y * fac
        return Box(min_x, min_y, max_x, max_y, dim_x, dim_y)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def write_bounds_2d(scene, cam_ob, me_ob, cur_frame):
    bpy.context.scene.frame_set(cur_frame)
    box = camera_view_bounds_2d(scene, cam_ob, me_ob).to_tuple()
    if np.count_nonzero(np.array(box)) != 0:
        return box
    else:
        return None


def normalize(data, frame_width, frame_height):
    """
    Normalize the bounding box coordinates relative to the frame size.

    :param data: The bounding box coordinates [x, y, width, height].
    :param frame_width: The width of the frame.
    :param frame_height: The height of the frame.
    :return: The normalized bounding box coordinates [x_center, y_center, width, height].
    """
    x_center = (data[0] + data[2] / 2) / frame_width
    y_center = (data[1] + data[3] / 2) / frame_height
    width = data[2] / frame_width
    height = data[3] / frame_height
    return x_center, y_center, width, height


def main(context, project_dir: str, frame_start: int, frame_end: int, tilt_angle: float, altitude: float, FOV: float):
    """
    Args:
        context (bpy.types.Context): context
        project_dir (str): project directory. Renders will be saved in <project_dir>/renders, labels - in <project_dir>/labels
        frame_start (int): number of the first frame
        frame_end (int): number of the last frame
        tilt_angle (float): camera tilt angle relative to the vertical. 0 - straight down, 90 - to the horizon
        altitude (float): flight altitude
        FOV (float): camera FOV
    """

    scene_renders_dir = os.path.join(project_dir, "images")
    labels_dir = os.path.join(project_dir, "labels")
    meta_path = os.path.join(project_dir, "meta.json")

    bpy.context.view_layer.objects.active = None

    if not os.path.isdir(project_dir):
        os.mkdir(project_dir)

    if os.path.isdir(scene_renders_dir):
        shutil.rmtree(scene_renders_dir)
    if os.path.isdir(labels_dir):
        shutil.rmtree(labels_dir)
    os.mkdir(scene_renders_dir)
    os.mkdir(labels_dir)
    scene_renders_dir = os.path.join(scene_renders_dir, "frame_")

    camera = bpy.data.objects['Camera']  # bpy.context.object
    camera.data.angle = np.deg2rad(FOV)
    camera.data.clip_end = 999999986991104

    # Set up the scene
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.camera = camera

    # Create Flight Path
    if "FlightPath" in bpy.data.objects:  # check if the flight path already exists
        obj = bpy.data.objects["FlightPath"]
        bpy.context.view_layer.objects.active = obj
        obj_collection = obj.users_collection[0]
        obj_collection.objects.unlink(obj)
        bpy.data.objects.remove(obj)

    # Create Flight Path - Copy and move bezier curve
    sphere_path = bpy.data.objects['BezierCurve']
    bpy.context.view_layer.objects.active = sphere_path
    sphere_path.select_set(True)
    bpy.ops.object.duplicate(linked=False)
    sphere_path.select_set(False)

    flight_path = bpy.context.view_layer.objects.active

    flight_path.location.z = flight_path.location.z + altitude
    flight_path.name = "FlightPath"
    bpy.context.view_layer.objects.active = None

    # estimate target offset from tilt
    depsgraph = bpy.context.evaluated_depsgraph_get()
    path_length = sum(s.calc_length() for s in sphere_path.evaluated_get(depsgraph).data.splines)
    tilt_tang = np.tan(np.deg2rad(tilt_angle))
    target_offset_meters = tilt_tang * altitude  # altitude
    relative_target_follow_offset = target_offset_meters / path_length
    relative_target_follow_offset = min(relative_target_follow_offset, 0.999)
    relative_target_follow_offset = max(relative_target_follow_offset, 0.001)

    # get camera
    context.scene.camera = camera

    # Clear old camera constraints
    for k in camera.constraints.keys():
        camera.constraints.remove(camera.constraints[k])
    camera.animation_data_clear()

    # make camera move along curve
    constraint = camera.constraints.new(type='FOLLOW_PATH')
    constraint.target = flight_path
    constraint.use_fixed_location = True
    constraint.use_curve_follow = True

    constraint.offset_factor = 0.0
    constraint.keyframe_insert(data_path="offset_factor", frame=1)
    constraint.offset_factor = relative_target_follow_offset
    constraint.keyframe_insert(data_path="offset_factor", frame=frame_end)

    # add camera target to the sphere
    if "SurfSphere" in bpy.data.objects:  # check if the sphere already exists
        sphere = bpy.data.objects["SurfSphere"]
        sphere.animation_data_clear()
    else:
        location = (0, 0, 0)
        radius = 1.0
        main_collection = bpy.data.collections['Collection']
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
        sphere = bpy.context.object
        bpy.ops.collection.objects_remove_all()
        main_collection.objects.link(sphere)
        sphere.name = "SurfSphere"
        sphere.animation_data_clear()
        sphere.select_set(False)

    sphere.hide_render = True
    for k in sphere.constraints.keys():
        sphere.constraints.remove(sphere.constraints[k])
    constraint_target = camera.constraints.new(type='TRACK_TO')
    constraint_target.target = sphere

    # make target sphere move
    constraint_sphere = sphere.constraints.new(type='FOLLOW_PATH')
    constraint_sphere.target = sphere_path
    constraint_sphere.use_fixed_location = True
    constraint_sphere.use_curve_follow = True

    constraint_sphere.offset_factor = relative_target_follow_offset
    constraint_sphere.keyframe_insert(data_path="offset_factor", frame=1)
    constraint_sphere.offset_factor = 1.0
    constraint_sphere.keyframe_insert(data_path="offset_factor", frame=frame_end)

    # config renderer
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = frame_end

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = scene_renders_dir

    # start render
    bpy.ops.render.render(animation=True)

    # make sphere visible (just in case)
    sphere.hide_render = False

    bpy.context.view_layer.objects.active = None

    # get labeled objects names
    google_collection = bpy.data.collections.get("Google 3D Tiles")
    labeled_objects_names = []
    for obj in google_collection.objects:
        if obj.name.startswith("obj"):
            labeled_objects_names.append(obj.name)

    frame_width = scene.render.resolution_x
    frame_height = scene.render.resolution_y

    for frame_current in range(frame_start, frame_end):
        all_data = ""

        # iterate over each labeled object
        for label in labeled_objects_names:
            me_ob = bpy.data.objects.get(label)

            data = write_bounds_2d(scene, camera, me_ob, frame_current)

            if data:
                x_center, y_center, width, height = normalize(data, frame_width, frame_height)
                all_data += f"0 {x_center} {y_center} {width} {height}\n"
            scene.frame_set(frame_current)

        # save label
        frame_current_str = str(frame_current).zfill(4)
        label_filepath = os.path.join(labels_dir, f'frame_{frame_current_str}.txt')
        with open(label_filepath, 'w') as data:
            data.write(str(all_data))

    objects_count = len(labeled_objects_names)
    data = {
        "altitude": altitude,
        "tilt_angle": tilt_angle,
        "objects_count": objects_count,
        "path_length": path_length,
        "frame_start": frame_start,
        "frame_end": frame_end
    }
    with open(meta_path, 'w') as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    # call from blender "Scripting" tab
    if bpy.context.space_data is not None and bpy.context.space_data.type == "TEXT_EDITOR":
        project_dir = default_config["project_dir"]
        frame_start = default_config['frame_start']
        frame_end = default_config['frame_end']
        tilt_angle = default_config['tilt_angle']
        altitude = default_config['altitude']
        FOV = default_config['FOV']

    # call from batch mode
    else:
        parser = argparse.ArgumentParser(description="Render and label images with Blender.")
        parser.add_argument("--project_dir", type=str, default=default_config["project_dir"],
                            help="Project directory. Renders will be saved in <project_dir>/renders, labels are saved in <project_dir>/labels")
        parser.add_argument("--frame_start", type=int, default=default_config["frame_start"],
                            help="Number of the first frame")
        parser.add_argument("--frame_end", type=int, default=default_config["frame_end"],
                            help="Number of the last frame")
        parser.add_argument("--tilt_angle", type=float, default=default_config["tilt_angle"],
                            help="Camera tilt angle relative to the vertical. 0 = straight down, 90 = horizon")
        parser.add_argument("--altitude", type=float, default=default_config["altitude"],
                            help="Flight altitude")
        parser.add_argument("--FOV", type=float, default=default_config["FOV"],
                            help="Camera FOV")
        args = parser.parse_args()

        project_dir = args.project_dir
        frame_start = args.frame_start
        frame_end = args.frame_end
        tilt_angle = args.tilt_angle
        altitude = args.altitude
        FOV = args.FOV

    os.makedirs(project_dir, exist_ok=True)

    old_stdout = sys.stdout
    log_file = open(os.path.join(project_dir, "message.log"), "w")
    sys.stdout = log_file

    main(bpy.context, project_dir, frame_start, frame_end, tilt_angle, altitude, FOV)

    sys.stdout = old_stdout
    log_file.close()
