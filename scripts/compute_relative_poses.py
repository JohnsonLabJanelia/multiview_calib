# ----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2020
# --------------------------------------------------------------------------
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


def main(
    setup="setup.json",
    intrinsics="intrinsics.json",
    landmarks="landmarks.json",
    landmarks_global="landmarks_global.json",
    filenames="filenames.json",
    method="8point",
    dump_images=True,
    th=20,
    output_path="output/relative_poses/",
):
    utils.mkdir(output_path)
    utils.config_logger(os.path.join(output_path, "relative_poses.log"))

    setup = utils.json_read(setup)
    intrinsics = utils.json_read(intrinsics)
    landmarks = utils.json_read(landmarks)
    landmarks_global = utils.json_read(landmarks_global)

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


if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup",
        "-s",
        type=str,
        required=True,
        default="setup.json",
        help="JSON file containing the camera setup",
    )
    parser.add_argument(
        "--intrinsics",
        "-i",
        type=str,
        required=True,
        default="intrinsics.json",
        help="JSON file containing the intrinsics parameters",
    )
    parser.add_argument(
        "--landmarks",
        "-l",
        type=str,
        required=True,
        default="landmarks.json",
        help="JSON file containing the landmark for each view",
    )
    parser.add_argument(
        "--landmarks_global",
        "-lg",
        type=str,
        required=True,
        default="landmarks_global.json",
        help="JSON file containing global landmark for each view",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        required=False,
        default="8point",
        help="Method to compute fundamental matrix: '8point', 'lmeds' or 'ransac'",
    )
    parser.add_argument(
        "--filenames",
        "-f",
        type=str,
        required=False,
        default="filenames.json",
        help="JSON file containing one filename of an image for each view. Used onyl if --dump_images is on",
    )
    parser.add_argument(
        "--dump_images",
        "-d",
        default=False,
        const=True,
        action="store_const",
        help="Saves images for visualisation",
    )
    parser.add_argument(
        "--th",
        "-th",
        type=int,
        required=False,
        default=20,
        help="Threshold for RANSAC method",
    )

    args = parser.parse_args()

    main(**vars(args))
