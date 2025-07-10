import numpy as np
import cv2
import argparse
import os
from scipy.io import savemat
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_path", required=True, type=str)
parser.add_argument("--width", required=True, type=int)
parser.add_argument("--height", required=True, type=int)
parser.add_argument("-sl", "--square_length", required=True, type=float)
parser.add_argument("-ml", "--marker_length", required=True, type=float)
parser.add_argument("-d","--dictionary", required=True, type=int)
parser.add_argument("-o", "--output_dir", required=True, type=str)

args = parser.parse_args()

img_path = args.img_path
dictionary = args.dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
board_size = (args.width, args.height)
board = cv2.aruco.CharucoBoard(board_size, args.square_length, args.marker_length, aruco_dict)
charuco_detector = cv2.aruco.CharucoDetector(board)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.00001)

images = []
for f in os.listdir(img_path):
    _, extension = os.path.splitext(f)
    if extension.lower() in [".jpg", ".jpeg", ".bmp", ".tiff", ".png", ".gif"]:
        images.append(f)
if len(images) == 0:
    print("No images found.")

cam_names = ["2012630", "2012631", "2012853", "2012855", "2012857", "2012862"]
images_all_cam = {}
for cam in cam_names:
    images_per_cam = []
    for image in images:
        this_image_cam_name = image.split("_")[0]
        if this_image_cam_name == cam:
            images_per_cam.append(os.path.join(img_path, image))
    images_all_cam[cam] = images_per_cam

#all_img_points = []
valid_imgs = []
valid_imgs_dict = {}
for cam in cam_names:
    img_file_list = images_all_cam[cam]
    valid_img_per_cam = []
    for img_file in img_file_list:
        frame = cv2.imread(img_file)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = (charuco_detector.detectBoard(frame_gray))   
        if charuco_corners is not None and charuco_ids is not None:
            if len(charuco_ids) > 6:
                img_id = img_file.split("/")[-1].split("_")[-1].split(".")[0]
                valid_img_per_cam.append(img_id)
    valid_imgs.append(valid_img_per_cam)
    valid_imgs_dict[cam] = valid_img_per_cam

# Convert to sets and find intersection
common_elements = set(valid_imgs[0]).intersection(*valid_imgs[1:])
print(common_elements)  # Output: {2, 3}
print(len(common_elements))

output_dir = args.output_dir
for cam in cam_names:
    per_cam_folder = os.path.join(output_dir, cam)
    print(per_cam_folder)
    os.makedirs(per_cam_folder, exist_ok=True)
    all_img_points = []
    all_obj_points = []
    print(len(valid_imgs_dict[cam]))
    for img_id in valid_imgs_dict[cam]:
        file_name = "{}_{}.jpg".format(cam, img_id)
        src_path = os.path.join(img_path, file_name)
        shutil.copy(src_path, per_cam_folder)

        frame = cv2.imread(src_path)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = (charuco_detector.detectBoard(frame_gray))   
        if charuco_corners is not None and charuco_ids is not None:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            all_img_points.append(img_points[:, 0])
            all_obj_points.append(obj_points[:, 0, :2])

    output_file = os.path.join(per_cam_folder, 'detect_pts.mat')
    savemat(output_file, {"my_img_pts": all_img_points, "my_obj_pts": all_obj_points})
