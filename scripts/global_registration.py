import numpy as np
import argparse
import matplotlib
import imageio
import logging
import cv2
import os
import warnings

warnings.filterwarnings("ignore")

matplotlib.use("Agg")

from multiview_calib import utils
from multiview_calib.extrinsics import (
    global_registration,
    visualise_global_registration,
)
from multiview_calib.bundle_adjustment_scipy import error_measure

logger = logging.getLogger(__name__)


def main(
    config,
    dump_images=True,
):
    root_folder = os.path.dirname(config)
    output_path = root_folder + "/output/global_registration"
    utils.mkdir(output_path)
    utils.config_logger(os.path.join(output_path, "global_registration.log"))

    setup = utils.json_read(root_folder + "/output/setup.json")
    ba_poses = utils.json_read(root_folder + "/output/bundle_adjustment/ba_poses.json")
    ba_points = utils.json_read(
        root_folder + "/output/bundle_adjustment/ba_points.json"
    )
    landmarks = utils.json_read(root_folder + "/output/landmarks.json")
    landmarks_global = utils.json_read(root_folder + "/output/landmarks_global.json")

    global_poses, global_triang_points = global_registration(
        ba_poses, ba_points, landmarks_global
    )
    if dump_images:
        filenames = utils.json_read(root_folder + "/output/filenames.json")
        visualise_global_registration(
            global_poses,
            landmarks_global,
            ba_poses,
            ba_points,
            filenames,
            output_path=output_path,
        )

    utils.json_write(os.path.join(output_path, "global_poses.json"), global_poses)
    utils.json_write(
        os.path.join(output_path, "global_triang_points.json"), global_triang_points
    )

    avg_dist, std_dist, median_dist = error_measure(
        setup,
        landmarks,
        global_poses,
        global_triang_points,
        scale=1,
        view_limit_triang=5,
    )
    logging.info("Per pair of view average error:")
    logging.info(
        "\t mean+-std: {:0.3f}+-{:0.3f} [unit of destination (dst) point set]".format(
            avg_dist, std_dist
        )
    )
    logging.info(
        "\t median:    {:0.3f}        [unit of destination (dst) point set]".format(
            median_dist
        )
    )

    r_x_c_180 = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    r_z_c_90 = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r_t = r_z_c_90 @ r_x_c_180

    rig_space_path = root_folder + "/calibration"
    utils.mkdir(rig_space_path)

    for key, value in global_poses.items():
        output_file = rig_space_path + "/Cam{}.yaml".format(key)

        new_r = np.asarray(value["R"]) @ (r_t.T)
        new_t = np.asarray(value["t"])

        cam_img = cv2.imread(filenames[key])
        h, w, _ = cam_img.shape

        utils.save_extrinsics_yaml(
            output_file,
            [w, h],
            np.asarray(value["K"]),
            np.asarray(value["dist"]),
            new_r,
            new_t,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)

    parser.add_argument(
        "--dump_images",
        "-d",
        default=True,
        const=True,
        action="store_const",
        help="Saves images for visualisation",
    )
    args = parser.parse_args()
    main(**vars(args))
