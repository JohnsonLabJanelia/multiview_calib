from multiview_calib import utils
import os
import argparse
import numpy as np
import pickle
from multiview_calib.extrinsics import verify_landmarks

parser = argparse.ArgumentParser()
parser.add_argument("-root_folder", "-r", type=str, required=True)

args = parser.parse_args()
root_folder = args.root_folder

config = utils.json_read(root_folder + "/config.json")
img_path = config["img_path"]
cam_ordered = config["cam_ordered"]
world_coordinate_img = config["world_coordinate_img"]
first_view = config["first_view"]
second_view_order = config["second_view_order"]

minimal_tree = []
for second_view_idx in second_view_order:
    minimal_tree.append([cam_ordered[first_view], cam_ordered[second_view_idx]])

setup_dict = {"views": cam_ordered, "minimal_tree": minimal_tree}
utils.json_write(root_folder + "/output/setup.json", setup_dict)

filenames = {}
for cam in cam_ordered:
    filename = "_".join([cam, world_coordinate_img])
    filenames[cam] = os.path.join(img_path, filename + ".tiff")

utils.json_write(root_folder + "/output/filenames.json", filenames)

intrinsics = {}
for cam in cam_ordered:
    cam_intrinsics_file = root_folder + "/output/intrinsics/{}.yaml".format(cam)
    ret, img_size, cam_matrix, dist_coefficients = utils.read_intrinsics_yaml(
        cam_intrinsics_file
    )
    dist_coefficients = np.squeeze(dist_coefficients)
    intrinsics_per_cam_dict = {
        "K": cam_matrix.tolist(),
        "dist": dist_coefficients.tolist(),
    }
    intrinsics[cam] = intrinsics_per_cam_dict


utils.json_write(root_folder + "/output/intrinsics.json", intrinsics)


charuco_config = config["charuco_setup"]
width = charuco_config["w"]
height = charuco_config["h"]

all_cams = []
for f in os.listdir(root_folder + "/output/intrinsics"):
    if f.endswith(".yaml"):
        all_cams.append(f.split(".")[0])

landmarks = {}
for cam in all_cams:
    landmarks_file = root_folder + "/output/intrinsics/landmarks_{}.pkl".format(cam)
    with open(landmarks_file, "rb") as f:
        landmarks_per_cam_dict = pickle.load(f)
    landmarks[cam] = landmarks_per_cam_dict

all_img_names = []
for key, value in landmarks.items():
    all_img_names.extend(list(value.keys()))
unique_names, counts = np.unique(all_img_names, return_counts=True)

# assign a unique id for each image, and each corner
img_id = 0
landmarks_final = {}
landmarks_global = {}
for i, img_name in enumerate(unique_names):
    if counts[i] > 1:
        for cam in landmarks.keys():
            per_cam_ids = []
            per_cam_landmarks = []
            per_cam_global_ids = []

            if img_name in landmarks[cam]:
                marker_dict = landmarks[cam][img_name]

                for j in range(len(marker_dict["ids"])):
                    point_unique_id = img_id * (width - 1) * (height - 1) + int(
                        marker_dict["ids"][j][0]
                    )
                    per_cam_ids.append(point_unique_id)
                    per_cam_landmarks.append(marker_dict["corners"][j][0].tolist())

                    if img_name == world_coordinate_img + ".tiff":
                        per_cam_global_ids.append(point_unique_id)

                if cam in landmarks_final.keys():
                    landmarks_final[cam]["ids"].extend(per_cam_ids)
                    landmarks_final[cam]["landmarks"].extend(per_cam_landmarks)
                else:
                    landmarks_final[cam] = {
                        "ids": per_cam_ids,
                        "landmarks": per_cam_landmarks,
                    }

                if img_name == world_coordinate_img + ".tiff":
                    landmarks_global[cam] = {
                        "ids": per_cam_global_ids,
                        "landmarks_global": np.squeeze(
                            marker_dict["objpoints"]
                        ).tolist(),
                    }
        img_id = img_id + 1


res, msg = verify_landmarks(landmarks_final)
if not res:
    raise ValueError(msg)
utils.json_write(root_folder + "/output/landmarks.json", landmarks_final)

max_number_ids = 0
cam_with_max_number_ids = next(iter(landmarks_global.keys()))
for cam, marker in landmarks_global.items():
    if len(marker["ids"]) > max_number_ids:
        cam_with_max_number_ids = cam
        max_number_ids = len(marker["ids"])

utils.json_write(
    root_folder + "/output/landmarks_global.json",
    landmarks_global[cam_with_max_number_ids],
)
