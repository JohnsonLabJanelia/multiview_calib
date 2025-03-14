import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imageio
import logging
import time
import cv2
import os
import warnings

warnings.filterwarnings("ignore")

from multiview_calib import utils
from multiview_calib.bundle_adjustment_scipy import (
    build_input,
    bundle_adjustment,
    evaluate,
    visualisation,
    unpack_camera_params,
)
from multiview_calib.singleview_geometry import reprojection_error
from multiview_calib.extrinsics import verify_view_tree, verify_landmarks

logger = logging.getLogger(__name__)

"""
ba_config = {
    "each_training": 1,  # to use less datatpoint during the optimization
    "each_visualisation": 1,  # to use less datatpoints in the visualisation
    "optimize_camera_params": True,
    "optimize_points": True,
    "ftol": 1e-8,
    "xtol": 1e-8,
    "loss": "linear",
    "f_scale": 1,
    "max_nfev": 200,  # first optimization
    "max_nfev2": 200,  # second optimization after outlier removal
    "bounds": True,
    "bounds_cp": [0.3] * 3 + [1] * 3 + [10, 10, 10, 10] + [0.01, 0.01, 0, 0, 0],
    "bounds_pt": [100] * 3,
    "th_outliers_early": 1000,
    "th_outliers": 50,  # value in pixels defining a point to be an outlier. If None, do not remove outliers.
}
"""


def main(root_folder, iter1=200, iter2=200, dump_images=True):
    config = utils.json_read(root_folder + "/config.json")
    ba_config = config["ba_config"]
    if iter1 is not None:
        ba_config["max_nfev"] = iter1
    if iter2 is not None:
        ba_config["max_nfev2"] = iter2

    output_path = root_folder + "/output/bundle_adjustment"
    utils.mkdir(output_path)
    utils.config_logger(os.path.join(output_path, "bundle_adjustment.log"))
    setup = utils.json_read(root_folder + "/output/setup.json")
    intrinsics = utils.json_read(root_folder + "/output/intrinsics.json")
    extrinsics = utils.json_read(root_folder + "/output/relative_poses/poses.json")
    landmarks = utils.json_read(root_folder + "/output/landmarks.json")
    filenames_images = utils.json_read(root_folder + "/output/filenames.json")

    n_dist_coeffs = len(list(intrinsics.values())[0]["dist"])

    if not verify_view_tree(setup["minimal_tree"]):
        raise ValueError("minimal_tree is not a valid tree!")

    res, msg = verify_landmarks(landmarks)
    if not res:
        raise ValueError(msg)

    views = setup["views"]
    logging.info("-" * 20)
    logging.info("Views: {}".format(views))

    if ba_config["each_training"] < 2 or ba_config["each_training"] is None:
        logging.info("Use all the landmarks.")
    else:
        logging.info(
            "Subsampling the landmarks to 1 every {}.".format(
                ba_config["each_training"]
            )
        )

    logging.info(
        "Preparing the input data...(this can take a while depending on the number of points to triangulate)"
    )
    start = time.time()
    (
        camera_params,
        points_3d,
        points_2d,
        camera_indices,
        point_indices,
        n_cameras,
        n_points,
        ids,
        views_and_ids,
    ) = build_input(
        views,
        intrinsics,
        extrinsics,
        landmarks,
        each=ba_config["each_training"],
        view_limit_triang=4,
    )
    logging.info(
        "The preparation of the input data took: {:0.2f}s".format(time.time() - start)
    )
    logging.info("Sizes:")
    logging.info("\t camera_params: {}".format(camera_params.shape))
    logging.info("\t points_3d: {}".format(points_3d.shape))
    logging.info("\t points_2d: {}".format(points_2d.shape))

    f0 = evaluate(
        camera_params,
        points_3d,
        points_2d,
        camera_indices,
        point_indices,
        n_cameras,
        n_points,
    )

    if dump_images:
        plt.figure()
        camera_indices_rav = np.vstack([camera_indices] * 2).T.ravel()
        for view_idx in range(n_cameras):
            m = np.where(camera_indices_rav == view_idx)[0]
            plt.plot(f0[m], label="{}".format(views[view_idx]))
        plt.title("Residuals at initialization")
        plt.ylabel("Residual [pixels]")
        plt.xlabel("X and Y coordinates")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(
            os.path.join(output_path, "initial_residuals.jpg"),
            bbox_inches="tight",
        )

    outliers = []
    if ba_config["th_outliers_early"] == 0 or ba_config["th_outliers_early"] is None:
        logging.info("No early outlier rejection.")
    else:
        logging.info("Early Outlier rejection:")
        logging.info("\t threshold outliers: {}".format(ba_config["th_outliers_early"]))

        f0_ = np.abs(f0.reshape(-1, 2))
        mask_outliers = np.logical_or(
            f0_[:, 0] > ba_config["th_outliers_early"],
            f0_[:, 1] > ba_config["th_outliers_early"],
        )

        utils.json_write(
            os.path.join(output_path, "outliers_early.json"),
            [
                views_and_ids[i] + (points_2d[i].tolist(),)
                for i, m in enumerate(mask_outliers)
                if m
            ],
        )

        point_indices = point_indices[~mask_outliers]
        camera_indices = camera_indices[~mask_outliers]
        points_2d = points_2d[~mask_outliers]
        views_and_ids = [views_and_ids[i] for i, m in enumerate(~mask_outliers) if m]
        optimized_points = np.int32(list(set(point_indices)))
        logging.info(
            "\t Number of points considered outliers: {}".format(sum(mask_outliers))
        )

        if sum(mask_outliers) / len(mask_outliers) > 0.5:
            logging.info("!" * 20)
            logging.info(
                "More than half of the data points have been considered outliers! Something may have gone wrong."
            )
            logging.info("!" * 20)

    if dump_images:
        f01 = evaluate(
            camera_params,
            points_3d,
            points_2d,
            camera_indices,
            point_indices,
            n_cameras,
            n_points,
        )

        plt.figure()
        camera_indices_rav = np.vstack([camera_indices] * 2).T.ravel()
        for view_idx in range(n_cameras):
            m = np.where(camera_indices_rav == view_idx)[0]
            plt.plot(f01[m], label="{}".format(views[view_idx]))
        plt.title("Residuals after early outlier rejection")
        plt.ylabel("Residual [pixels]")
        plt.xlabel("X and Y coordinates")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(
            os.path.join(output_path, "early_outlier_rejection_residuals.jpg"),
            bbox_inches="tight",
        )

    if ba_config["bounds"]:
        logging.info("Bounded optimization:")
        logging.info("\t LB(x)=x-bound; UB(x)=x+bound")
        logging.info("\t rvec bounds=({},{},{})".format(*ba_config["bounds_cp"][:3]))
        logging.info("\t tvec bounds=({},{},{})".format(*ba_config["bounds_cp"][3:6]))
        logging.info(
            "\t k bounds=(fx={},fy={},c0={},c1={})".format(
                *ba_config["bounds_cp"][6:10]
            )
        )
        if len(ba_config["bounds_cp"][10:]) == 5:
            logging.info(
                "\t dist bounds=({},{},{},{},{})".format(*ba_config["bounds_cp"][10:])
            )
        else:
            logging.info(
                "\t dist bounds=({},{},{},{},{},{},{},{})".format(
                    *ba_config["bounds_cp"][10:]
                )
            )
        logging.info(
            "\t 3d points bounds=(x={},y={},z={})".format(*ba_config["bounds_pt"])
        )
    else:
        logging.info("Unbounded optimization.")

    logging.info("Least-Squares optimization of 3D points and camera parameters:")
    logging.info("\t optimize camera parameters: {}".format(True))
    logging.info("\t optimize 3d points: {}".format(True))
    logging.info("\t ftol={:0.3e}".format(ba_config["ftol"]))
    logging.info("\t xtol={:0.3e}".format(ba_config["xtol"]))
    logging.info(
        "\t loss={} f_scale={:0.2f}".format(ba_config["loss"], ba_config["f_scale"])
    )
    logging.info("\t max_nfev={}".format(ba_config["max_nfev"]))

    new_camera_params, new_points_3d = bundle_adjustment(
        camera_params,
        points_3d,
        points_2d,
        camera_indices,
        point_indices,
        n_cameras,
        n_points,
        ids,
        optimize_camera_params=ba_config["optimize_camera_params"],
        optimize_points=ba_config["optimize_points"],
        ftol=ba_config["ftol"],
        xtol=ba_config["xtol"],
        loss=ba_config["loss"],
        f_scale=ba_config["f_scale"],
        max_nfev=ba_config["max_nfev"],
        bounds=ba_config["bounds"],
        bounds_cp=ba_config["bounds_cp"],
        bounds_pt=ba_config["bounds_pt"],
        verbose=True,
        eps=1e-12,
        n_dist_coeffs=n_dist_coeffs,
    )

    f1 = evaluate(
        new_camera_params,
        new_points_3d,
        points_2d,
        camera_indices,
        point_indices,
        n_cameras,
        n_points,
    )

    avg_abs_res = np.abs(f1[:]).mean()
    logging.info(
        "Average absolute residual: {:0.2f} over {} points.".format(
            avg_abs_res, len(f1) / 2
        )
    )
    if avg_abs_res > 15:
        logging.info("!" * 20)
        logging.info(
            "The average absolute residual error is high! Something may have gone wrong.".format(
                avg_abs_res
            )
        )
        logging.info("!" * 20)

    if dump_images:
        plt.figure()
        camera_indices_rav = np.vstack([camera_indices] * 2).T.ravel()
        for view_idx in range(n_cameras):
            m = np.where(camera_indices_rav == view_idx)[0]
            plt.plot(f1[m], label="{}".format(views[view_idx]))
        plt.title("Residuals after optimization")
        plt.ylabel("Residual [pixels]")
        plt.xlabel("X and Y coordinates")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(
            os.path.join(output_path, "optimized_residuals.jpg"),
            bbox_inches="tight",
        )

    # Find ouliers points and remove them form the optimization.
    # These might be the result of inprecision in the annotations.
    # in this case we remove the resduals higher than 20 pixels.
    if ba_config["th_outliers"] == 0 or ba_config["th_outliers"] is None:
        logging.info("No outlier rejection.")
    else:
        logging.info("Outlier rejection:")
        logging.info("\t threshold outliers: {}".format(ba_config["th_outliers"]))
        logging.info("\t max_nfev={}".format(ba_config["max_nfev2"]))

        f1_ = np.abs(f1.reshape(-1, 2))
        mask_outliers = np.logical_or(
            f1_[:, 0] > ba_config["th_outliers"], f1_[:, 1] > ba_config["th_outliers"]
        )

        utils.json_write(
            os.path.join(output_path, "outliers_optimized.json"),
            [
                views_and_ids[i] + (points_2d[i].tolist(),)
                for i, m in enumerate(mask_outliers)
                if m
            ],
        )

        point_indices = point_indices[~mask_outliers]
        camera_indices = camera_indices[~mask_outliers]
        points_2d = points_2d[~mask_outliers]
        views_and_ids = [views_and_ids[i] for i, m in enumerate(~mask_outliers) if m]
        optimized_points = np.int32(list(set(point_indices)))
        logging.info(
            "\t Number of points considered outliers: {}".format(sum(mask_outliers))
        )

        if sum(mask_outliers) == 0:
            logging.info("\t Exit.")
        else:
            if sum(mask_outliers) / len(mask_outliers) > 0.5:
                logging.info("!" * 20)
                logging.info(
                    "More than half of the data points have been considered outliers! Something may have gone wrong."
                )
                logging.info("!" * 20)

            logging.info("\t New sizes:")
            logging.info("\t\t camera_params: {}".format(camera_params.shape))
            logging.info("\t\t points_3d: {}".format(points_3d.shape))
            logging.info("\t\t points_2d: {}".format(points_2d.shape))

            if len(points_2d) == 0:
                logging.info("No points left! Exit.")
                return

            new_camera_params, new_points_3d = bundle_adjustment(
                camera_params,
                points_3d,
                points_2d,
                camera_indices,
                point_indices,
                n_cameras,
                n_points,
                ids,
                optimize_camera_params=ba_config["optimize_camera_params"],
                optimize_points=ba_config["optimize_points"],
                ftol=ba_config["ftol"],
                xtol=ba_config["xtol"],
                loss=ba_config["loss"],
                f_scale=ba_config["f_scale"],
                max_nfev=ba_config["max_nfev2"],
                bounds=ba_config["bounds"],
                bounds_cp=ba_config["bounds_cp"],
                bounds_pt=ba_config["bounds_pt"],
                verbose=True,
                eps=1e-12,
                n_dist_coeffs=n_dist_coeffs,
            )

            f2 = evaluate(
                new_camera_params,
                new_points_3d,
                points_2d,
                camera_indices,
                point_indices,
                n_cameras,
                n_points,
            )

            avg_abs_res = np.abs(f2[:]).mean()
            logging.info(
                "Average absolute residual: {:0.2f} over {} points.".format(
                    avg_abs_res, len(f2) / 2
                )
            )
            if avg_abs_res > 15:
                logging.info("!" * 20)
                logging.info(
                    "The average absolute residual error (after outlier removal) is high! Something may have gone wrong.".format(
                        avg_abs_res
                    )
                )
                logging.info("!" * 20)

            if dump_images:
                plt.figure()
                camera_indices_rav = np.vstack([camera_indices] * 2).T.ravel()
                for view_idx in range(n_cameras):
                    m = np.where(camera_indices_rav == view_idx)[0]
                    plt.plot(f2[m], label="{}".format(views[view_idx]))
                plt.title("Residuals after outlier removal")
                plt.ylabel("Residual [pixels]")
                plt.xlabel("X and Y coordinates")
                plt.legend()
                plt.grid()
                plt.show()
                plt.savefig(
                    os.path.join(
                        output_path,
                        "optimized_residuals_outliers_removal.jpg",
                    ),
                    bbox_inches="tight",
                )

    logging.info("Reprojection errors (mean+-std pixels):")
    ba_poses = {}
    for i, (view, cp) in enumerate(zip(views, new_camera_params)):
        K, R, t, dist = unpack_camera_params(cp)
        ba_poses[view] = {
            "R": R.tolist(),
            "t": t.tolist(),
            "K": K.tolist(),
            "dist": dist.tolist(),
        }

        points3d = new_points_3d[point_indices[camera_indices == i]]
        points2d = points_2d[camera_indices == i]

        if len(points3d) == 0:
            raise RuntimeError("All 3D points have been discarded/considered outliers.")

        mean_error, std_error = reprojection_error(R, t, K, dist, points3d, points2d)
        logging.info(
            "\t {} n_points={}: {:0.3f}+-{:0.3f}".format(
                view, len(points3d), mean_error, std_error
            )
        )

    logging.info("Reprojection errors (median pixels):")
    ba_poses = {}
    for i, (view, cp) in enumerate(zip(views, new_camera_params)):
        K, R, t, dist = unpack_camera_params(cp)
        ba_poses[view] = {
            "R": R.tolist(),
            "t": t.tolist(),
            "K": K.tolist(),
            "dist": dist.tolist(),
        }

        points3d = new_points_3d[point_indices[camera_indices == i]]
        points2d = points_2d[camera_indices == i]

        mean_error, std_error = reprojection_error(
            R, t, K, dist, points3d, points2d, "median"
        )
        logging.info(
            "\t {} n_points={}: {:0.3f}".format(view, len(points3d), mean_error)
        )

    ba_points = {
        "points_3d": new_points_3d[optimized_points].tolist(),
        "ids": np.array(ids)[optimized_points].tolist(),
    }

    if ba_config["each_visualisation"] < 2 or ba_config["each_visualisation"] is None:
        logging.info("Visualise all the annotations.")
    else:
        logging.info(
            "Subsampling the annotations to visualise to 1 every {}.".format(
                ba_config["each_visualisation"]
            )
        )

    path = output_path if dump_images else None
    visualisation(
        setup,
        landmarks,
        filenames_images,
        new_camera_params,
        new_points_3d,
        points_2d,
        camera_indices,
        each=ba_config["each_visualisation"],
        path=path,
    )

    utils.json_write(os.path.join(output_path, "ba_poses.json"), ba_poses)
    utils.json_write(os.path.join(output_path, "ba_points.json"), ba_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", "-r", type=str, required=True)

    parser.add_argument(
        "--dump_images",
        "-d",
        default=True,
        const=True,
        action="store_const",
        help="Saves images for visualisation",
    )

    parser.add_argument(
        "--iter1",
        "-it1",
        type=int,
        required=False,
        default=None,
        help="Maximum number of iterations of the first optimization",
    )
    parser.add_argument(
        "--iter2",
        "-it2",
        type=int,
        required=False,
        default=None,
        help="Maximum number of iterations of the second optimization after outlier rejection",
    )

    args = parser.parse_args()

    main(**vars(args))
