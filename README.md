# Multiple view Camera calibration tool
This tool allows to compute the intrinsic and extrinsic camera parameters of a set of synchronized cameras with overlapping field of view. The intrinsics estimation is based on the OpenCV's camera calibration framework and it is used on each camera separately. In the extrinsics estimation, an initial solution (extrinsic parameters) is computed first using a linear approach then refined using bundle adjustment.  The output are camera poses (intrinsic matrix, distortion parameters, rotations and translations) w.r.t. either the first camera or a global reference system.

## Prerequisites

- numpy
- scipy
- imageio
- matplotlib
- OpenCV

## Installation
```
cd MULTIVIEW_CALIB_MASTER
pip install .
```

## Usage

This repo is adapated from https://github.com/cvlab-epfl/multiview_calib.

We use charuco board for intrinsic parameters estimation, since it works better for bigger images. Make sure there is a config file. 

## Intrinsics estimation
1. Compute intrinsic parameters:
Run the following script:
```
python charuco_intrinsics.py -r ../examples/robot_02_11
```
The script outputs several useful information for debugging purposes. One of them is the per keypoint reprojection error, another the monotonicity of the distortion function. If the distortion function is not monotonic, we suggest to sample more precise points on the corner of the image first. If this is not enought, try the Rational Model (-rm) instead. The Rational Model is a model of the lens that is more adapted to cameras with wider lenses.
To furter understand if the calibration went well, you should perform a visual inspection of the undistorted images that have been saved. The lines in the images should be straight and the picture must look like a normal picture. In case of failure try to update Opencv or re-take the video/pictures.

## Extrinsics estimation
2. Compute relative poses:
To recover the pose of each one of the cameras in the rig w.r.t. the first camera we first compute relative poses between pairs of views and then concatenate them to form a tree. To do so, we have to manually define a minimal set of pairs of views that connect every camera. This is done in the file `setup.json`.
```
-Note: do not pair cameras that are facing each other! Recovering proper geometry in this specifc case is difficult.
```
The file named `landmarks.json` contains precise image points for each view that are used to compute fundamental matrices and poses. The file `ìntrinsics.json` contains the intrinsic parameters for each view that we have computed previously. The file `filenames.json` contains a filename of an image for each view which are is used for visualisation purposes.
Check section `Input files` for more details on the file formats.

```
python compute_relative_poses.py -r ../examples/robot_02_11
```
The result of this operation are relative poses up to scale (the translation vector is unit vector).

3. Format data 
```
python format_for_calibration.py -r ../examples/robot_02_11
```

4. Concatenate relative poses:
In this step we concatenate/chain all the relative poses to obtain an approximation of the actual camera poses. The poses are defined w.r.t the first camera. At every concatenation we scale the current relative pose to match the scale of the previous ones. This to have roughly the same scale for each camera.
The file `relative_poses.json` is the output of the previous step.
```
python concatenate_relative_poses.py -r ../examples/robot_02_11
```
5. Bundle adjustment:
Nonlinear Least squares refinement of intrinsic and extrinsic parameters and 3D points. The camera poses output of this step are up to scale.
The file `poses.json` is the output of the previous step (Concatenate relative poses).
```
python bundle_adjustment.py -r ../examples/robot_02_11 
```

#### Transformation to the global reference system:
The poses and 3D points computed using the bundle adjustment are all w.r.t. the first camera and up to scale.
In order to have the poses in the global/world reference system we have to estimate the rigid transformation between the two reference systems. To do so we perform a rigid allignement of the 3D points computed using bundle adjustment and their corresponding ones in global/world coordinate (at least 4 non-symmetric points). These must be defined in the file `landmarks_global.json` and have the same ID of the points defined in `landmarks.json`. Note that there is no need to specify the global coordinate for all landmarks defined in `landmarks.json`; a subset is enough. Given these correspondeces, the following command will find the best rigid transform in the least squares sense between the two point sets and then update the poses computed by the bundle adjustment. The output are the update poses saved in `global_poses.json`. NOTE: make sure the points used here are not symmetric nor close to be symmetric as this implies multiple solutions whcih is not handeled!
```
python global_registration.py -r ../examples/robot_02_11
```
If the global landmarks are a different set of points than the one used during the optimization, you can use the following command to compute the `ba_points.json`.

#### Check final results
```
python 17cams.py -i ../examples/robot_02_11/output/global_registration/rig_space 
```
