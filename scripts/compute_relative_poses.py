import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import imageio
import logging
import cv2
import os
import warnings

warnings.filterwarnings("ignore")

from multiview_calib import utils
from multiview_calib.extrinsics import (
    compute_relative_poses,
    visualise_epilines,
    verify_view_tree,
    verify_landmarks,
)

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument(
    "--th",
    "-th",
    type=int,
    required=False,
    default=20,
    help="Threshold for RANSAC method",
)
parser.add_argument(
    "--method",
    "-m",
    type=str,
    required=False,
    default="lmeds",
    help="Method to compute fundamental matrix: '8point', 'lmeds' or 'ransac'",
)
parser.add_argument("--dump_images", "-d", action="store_true")
args = parser.parse_args()

config_file = args.config
root_folder = os.path.dirname(config_file)
config = utils.json_read(config_file)
filenames = root_folder + "/output/filenames.json"
dump_images = args.dump_images
th = args.th
method = args.method
output_path = root_folder + "/output/relative_poses/"

utils.mkdir(output_path)
utils.config_logger(os.path.join(output_path, "relative_poses.log"))

setup = utils.json_read(root_folder + "/output/setup.json")
intrinsics = utils.json_read(root_folder + "/output/intrinsics.json")
landmarks = utils.json_read(root_folder + "/output/landmarks.json")
landmarks_global = utils.json_read(root_folder + "/output/landmarks_global.json")

if not verify_view_tree(setup["minimal_tree"]):
    raise ValueError("minimal_tree is not a valid tree!")

res, msg = verify_landmarks(landmarks)
if not res:
    raise ValueError(msg)

relative_poses = compute_relative_poses(
    setup["minimal_tree"], intrinsics, landmarks, method, th, verbose=2
)

if dump_images:
    visualise_epilines(
        setup["minimal_tree"],
        relative_poses,
        intrinsics,
        landmarks,
        landmarks_global,
        filenames,
        output_path=output_path,
    )

relative_poses = utils.dict_keys_to_string(relative_poses)
utils.json_write(os.path.join(output_path, "relative_poses.json"), relative_poses)
