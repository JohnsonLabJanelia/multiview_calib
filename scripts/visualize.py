import rerun as rr
import numpy as np
import argparse
import cv2
import os
from multiview_calib import utils


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
args = parser.parse_args()

print(args.config)
config_file = args.config
calibration_dir = os.path.dirname(config_file)
config = utils.json_read(config_file)
gt_pts = config["gt_pts"]

cam_ordered = config["cam_ordered"]

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


rr.init("calibration")
rr.spawn()

rr.set_time_sequence("stable_time", 0)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
if (len(cam_ordered)==17):
    rr.log("arena", rr.Boxes3D(centers=[0, 0, -174.6], half_sizes=[914.4, 914.4, 174.6]))
    rr.log("shelter", rr.Boxes3D(centers=[1014.4, 0, -174.6], half_sizes=[100, 100, 174.6]))
else:
    for key, value in gt_pts.items():
        rr.log(key, rr.Points3D(value))

for order, serial in enumerate(cam_ordered):
    ## load yaml file
    yaml_file_name = calibration_dir + "/calibration/Cam{}.yaml".format(serial)
    cam_params = load_yaml_file(yaml_file_name)

    if cam_params:
        # compute camera pose
        rotation = cam_params["rc_ext"].T
        translation = -np.matmul(rotation, cam_params["tc_ext"][:, 0])

        resolution = [cam_params["image_width"], cam_params["image_height"]]

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
