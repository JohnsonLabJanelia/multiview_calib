import rerun as rr
import numpy as np
import argparse
import cv2
import os
from multiview_calib import utils
import rerun.blueprint as rrb
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--calibration_dir", type=str, required=True)
parser.add_argument("-n", "--last_n", type=int, default=5)
args = parser.parse_args()
calibration_dir = args.calibration_dir

all_calib_dates = [f.path for f in os.scandir(calibration_dir) if f.is_dir()]
all_calib_dates.sort()
all_calib_dates = all_calib_dates[-args.last_n :]


def load_yaml_file(yaml_cam_name):
    cam_params = {}
    fs = cv2.FileStorage(yaml_cam_name, cv2.FILE_STORAGE_READ)
    if fs.isOpened():
        cam_params["image_width"] = int(fs.getNode("image_width").real())
        cam_params["image_height"] = int(fs.getNode("image_height").real())
        cam_params["camera_matrix"] = fs.getNode("camera_matrix").mat()
        cam_params["distortion_coefficients"] = fs.getNode(
            "distortion_coefficients"
        ).mat()
        cam_params["tc_ext"] = fs.getNode("tc_ext").mat()
        cam_params["rc_ext"] = fs.getNode("rc_ext").mat()
        return cam_params
    else:
        return False


rr.init("Calibration", spawn=True)

blueprint = rrb.Blueprint(
    rrb.Spatial3DView(
        origin="/",
        name="3D Scene",
        background=[0, 0, 0],  # RGB for black
    ),
    collapse_panels=True,
)

rr.send_blueprint(blueprint)
rr.set_time_sequence("stable_time", 0)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
rr.log(
    "arena",
    rr.Boxes3D(centers=[0, 0, -174.6], half_sizes=[914.4, 914.4, 174.6]),
)
rr.log(
    "shelter",
    rr.Boxes3D(centers=[1014.4, 0, -174.6], half_sizes=[100, 100, 174.6]),
)


date_days = []
for calib_folder in all_calib_dates:
    calib_date = calib_folder.split("/")[-1]
    calib_time = [int(split_str) for split_str in calib_date.split("_")]
    date_days.append(date(calib_time[0], calib_time[1], calib_time[2]))

date_days_relative = [(date_day - date_days[0]).days for date_day in date_days]
sequence_idx = 0
for idx, calib_folder in enumerate(all_calib_dates):
    calib_date = calib_folder.split("/")[-1]
    print(calib_folder)
    config_file = calib_folder + "/config.json"
    calibration_dir = os.path.dirname(config_file)
    config = utils.json_read(config_file)
    gt_pts = config["gt_pts"]
    cam_ordered = config["cam_ordered"]

    rr.set_time_sequence(
        "stable_time", date_days_relative[idx] * 3
    )  # slowdown 3 times

    for key, value in gt_pts.items():
        rr.log(key, rr.Points3D(value))

    for order, serial in enumerate(cam_ordered):
        ## load yaml file
        yaml_file_name = calib_folder + "/calibration/Cam{}.yaml".format(
            serial
        )
        cam_params = load_yaml_file(yaml_file_name)

        if cam_params:
            # compute camera pose
            rotation = cam_params["rc_ext"].T
            translation = -np.matmul(rotation, cam_params["tc_ext"][:, 0])

            resolution = [
                cam_params["image_width"],
                cam_params["image_height"],
            ]

            # rr.log("world/camera/{}_{}".format(order, calib_date), rr.Transform3D(translation=translation, mat3x3=rotation))
            rr.log(
                "world/camera/{}_{}".format(serial, order),
                rr.Transform3D(translation=translation, mat3x3=rotation),
            )

            rr.log(
                # "world/camera/{}_{}".format(order, calib_date),
                "world/camera/{}_{}".format(serial, order),
                rr.Pinhole(
                    resolution=resolution,
                    image_from_camera=cam_params["camera_matrix"],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )

    sequence_idx = sequence_idx + 1
