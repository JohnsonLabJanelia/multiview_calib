import cv2 as cv
import numpy as np
import os
import argparse
import pickle
import logging
from multiview_calib import utils
import colorsys
import concurrent.futures
from functools import partial
import matplotlib

matplotlib.use("Agg")  # Set the backend to 'Agg'
import matplotlib.pyplot as plt

from multiview_calib.intrinsics import probe_monotonicity

logger = logging.getLogger(__name__)


def generate_distinct_colors(n):
    """Generate `n` distinct RGB colors evenly spaced in HSV space."""
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 1, 1)) for i in range(n)
    ]


def read_chessboards(images, board, aruco_dict, number_of_markers, verbose):
    """
    Charuco base pose estimation.
    """
    all_corners = []
    all_ids = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.00001)
    if verbose:
        wait_time = 0
    else:
        wait_time = 200

    charuco_detector = cv.aruco.CharucoDetector(board)
    objpoints = []
    imgpoints = []

    frame_0 = cv.imread(images[0])
    imsize = frame_0.shape[:2]
    all_im_ids = []
    num_points_thres = 6
    marker_colors = generate_distinct_colors(number_of_markers)

    for im in images:
        if verbose:
            logging.info("=> Processing image {0}".format(im))
        frame = cv.imread(im)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict)
        if (
            len(corners) >= num_points_thres
        ):  ## DLT require at last 6 pointes 3D-2D correspondences
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                charuco_detector.detectBoard(frame)
            )

            if charuco_corners is not None and charuco_ids is not None:
                obj_points, img_points = board.matchImagePoints(
                    charuco_corners, charuco_ids
                )

                # SUB PIXEL DETECTION
                for corner in corners:
                    cv.cornerSubPix(
                        gray,
                        corner,
                        winSize=(3, 3),
                        zeroZone=(-1, -1),
                        criteria=criteria,
                    )

                res2 = cv.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if (
                    res2[1] is not None
                    and res2[2] is not None
                    and len(res2[1]) >= num_points_thres
                ):
                    im_name = im.split("/")[-1]
                    all_im_ids.append("_".join(im_name.split("_")[1:]))
                    all_corners.append(res2[1])
                    all_ids.append(res2[2])

                    objpoints.append(obj_points)
                    imgpoints.append(img_points)

                    if verbose:
                        image_copy = np.copy(frame)

                        for pts_idx in range(res2[1].shape[0]):
                            # breakpoint()
                            cv.circle(
                                image_copy,
                                (
                                    int(res2[1][pts_idx, 0, 0]),
                                    int(res2[1][pts_idx, 0, 1]),
                                ),
                                15,
                                marker_colors[pts_idx],
                                -1,
                            )

                            cv.putText(
                                image_copy,
                                str(int(res2[2][pts_idx][0])),
                                (
                                    int(res2[1][pts_idx, 0, 0]),
                                    int(res2[1][pts_idx, 0, 1]),
                                ),
                                cv.FONT_HERSHEY_SIMPLEX,
                                2,
                                marker_colors[pts_idx],
                                5,
                            )

                        image_resize = cv.resize(image_copy, (1604, 1100))
                        cv.imshow("{}".format(im), image_resize)
                        key = cv.waitKey(wait_time)
                        if key == ord("q"):
                            break
    return all_corners, all_ids, imsize, objpoints, imgpoints, all_im_ids


def calibrate_camera(board, all_corners, all_ids, imsize, cam_name):
    """
    Calibrates the camera using the dected corners.
    """
    flags = 0
    flags += cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_FIX_ASPECT_RATIO
    # flags += (
    #     cv.CALIB_USE_INTRINSIC_GUESS
    #     + cv.CALIB_FIX_ASPECT_RATIO
    #     + cv.CALIB_RATIONAL_MODEL
    # )

    if cam_name == "710038":
        focal_length_init = 1780
    else:
        focal_length_init = 2300

    cameraMatrixInit = np.array(
        [
            [focal_length_init, 0.0, imsize[1] / 2.0],
            [0.0, focal_length_init, imsize[0] / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    distCoeffsInit = np.zeros((5, 1))

    (
        ret,
        camera_matrix,
        distortion_coefficients0,
        rotation_vectors,
        translation_vectors,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors,
    ) = cv.aruco.calibrateCameraCharucoExtended(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9),
    )

    return (
        ret,
        camera_matrix,
        distortion_coefficients0,
        rotation_vectors,
        translation_vectors,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors,
    )


def get_charuco_intrinsics(
    cam_name,
    images,
    charuco_setup,
    output_path,
    verbose,
):
    """
    args:
    charuco_setup: json file
    images: list of path to images
    output_path: path of folder to save results

    charuco_setup:
    "w", int: Number of squares in X direction
    "h", int:Number of squares in Y direction
    "square_side_length", float
    "marker_side_length", float
    "dictionary",int:
        dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,  DICT_4X4_1000=3,
        DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, DICT_6X6_50=8,
        DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, DICT_7X7_100=13,
        DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16
    """
    width = charuco_setup["w"]
    height = charuco_setup["h"]
    square_len = charuco_setup["square_side_length"]
    marker_len = charuco_setup["marker_side_length"]
    dict = charuco_setup["dictionary"]

    aruco_dict = cv.aruco.getPredefinedDictionary(dict)
    board_size = (width, height)
    board = cv.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)

    number_of_markers = (width - 1) * (height - 1)

    all_corners, all_ids, imsize, objpoints, imgpoints, all_im_ids = read_chessboards(
        images, board, aruco_dict, number_of_markers, verbose
    )
    print(
        "==> Camera: {}, number of valid images {} in {} total images.".format(
            cam_name, len(all_im_ids), len(images)
        )
    )
    landmark = {}
    for i, im_id in enumerate(all_im_ids):
        landmark[im_id] = {
            "corners": all_corners[i],
            "ids": all_ids[i],
            "objpoints": objpoints[i],
        }
    with open(output_path + "/landmarks_{}.pkl".format(cam_name), "wb") as f:
        pickle.dump(landmark, f)

    if len(all_im_ids) > 0:
        (
            ret,
            mtx,
            dist,
            rvecs,
            tvecs,
            std_dev_intrisics,
            std_dev_extrinsics,
            per_view_errors,
        ) = calibrate_camera(board, all_corners, all_ids, imsize, cam_name)

        # add metrics
        def reprojection_error(mtx, distCoeffs, rvecs, tvecs):
            # print reprojection error
            reproj_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv.projectPoints(
                    objpoints[i], rvecs[i], tvecs[i], mtx, distCoeffs
                )
                reproj_error += cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(
                    imgpoints2
                )
            reproj_error /= len(objpoints)
            return reproj_error

        reproj_error = reprojection_error(mtx, dist, rvecs, tvecs)
        logging.info(
            "{} RMS Reprojection Error: {}, Total Reprojection Error: {}".format(
                cam_name, ret, reproj_error
            )
        )

        alpha = 0.95
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            mtx, dist, imsize, alpha, imsize, centerPrincipalPoint=False
        )

        grid_norm, is_monotonic = probe_monotonicity(
            mtx, dist, newcameramtx, imsize, N=100, M=100
        )
        if not np.all(is_monotonic):
            logging.info(
                "{}: The distortion function is not monotonous for alpha={:0.2f}! To fix this we suggest sampling more precise points on the corner of the image first.  If this is not enough, use the option Rational Camera Model which more adpated to wider lenses. ".format(
                    cam_name, alpha
                )
            )

        frame = cv.imread(images[0])
        plt.figure()
        plt.imshow(cv.undistort(frame, mtx, dist, None, newcameramtx))
        grid = (
            grid_norm * newcameramtx[[0, 1], [0, 1]][None]
            + newcameramtx[[0, 1], [2, 2]][None]
        )
        plt.plot(
            grid[is_monotonic, 0],
            grid[is_monotonic, 1],
            ".g",
            label="monotonic",
            markersize=1.5,
        )
        plt.plot(
            grid[~is_monotonic, 0],
            grid[~is_monotonic, 1],
            ".r",
            label="not monotonic",
            markersize=1.5,
        )
        plt.legend()
        plt.grid()
        plt.savefig(
            os.path.join(output_path, "monotonicity_{}.jpg".format(cam_name)),
            bbox_inches="tight",
        )
        plt.close()

        output_file = os.path.join(output_path, "{}.yaml".format(cam_name))
        utils.save_intrinsics_yaml(output_file, imsize[1], imsize[0], mtx, dist)
        return mtx, dist, newcameramtx


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument("--verbose", "-v", action="store_true")

args = parser.parse_args()

config_file = args.config
root_folder = os.path.dirname(config_file)
config = utils.json_read(config_file)
img_path = config["img_path"]
cam_names = config["cam_ordered"]
charuco_setup = config["charuco_setup"]
output_path = os.path.join(root_folder + "/output/intrinsics/")
if_serial = args.verbose

utils.mkdir(output_path)
utils.config_logger(os.path.join(output_path, "intrinsics.log"))

images = []
for f in os.listdir(img_path):
    _, extension = os.path.splitext(f)
    if extension.lower() in [".jpg", ".jpeg", ".bmp", ".tiff", ".png", ".gif"]:
        images.append(f)
if len(images) == 0:
    logging.info("No images found.")

images_all_cams = []
for cam in cam_names:
    images_per_cam = []
    for image in images:
        this_image_cam_name = image.split("_")[0]
        if this_image_cam_name == cam:
            images_per_cam.append(os.path.join(img_path, image))
    images_all_cams.append(images_per_cam)

if if_serial:
    for idx, cam in enumerate(cam_names):
        get_charuco_intrinsics(
            cam, images_all_cams[idx], charuco_setup, output_path, True
        )
else:
    num_workers=17
    # parallel the process
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        partial_func = partial(
            get_charuco_intrinsics,
            charuco_setup=charuco_setup,
            output_path=output_path,
            verbose=False,
        )

        futures = [
            executor.submit(partial_func, cam_name, images)
            for cam_name, images in zip(cam_names, images_all_cams)
        ]

        for future in concurrent.futures.as_completed(futures):
            pass  # We don't store results, just wait for completion

    logging.info("All tasks completed.")
