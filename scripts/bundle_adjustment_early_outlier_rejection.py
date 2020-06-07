import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import imageio
import time
import cv2
import os

matplotlib.use("Agg")

from multiview_calib import utils 
from multiview_calib.bundle_adjustment_scipy import (build_input, bundle_adjustment, evaluate, 
                                                     visualisation, unpack_camera_params)

__config__ = {
    "each_training":1, # to use less datatpoint during the optimization
    "each_visualisation":1, # to use less datatpoints in the visualisation
    "optimize_camera_params":True, 
    "optimize_points":True, 
    "ftol":1e-8,
    "xtol":1e-8,
    "loss":"linear",
    "f_scale":1,
    "max_nfev":200, # first optimization
    "max_nfev2":200,# second optimization after outlier removal
    "bounds":True, 
    "bounds_cp":[0.3]*3+[1]*3+[10,10,10,10]+[0,0,0,0,0],
    "bounds_pt":[100]*3,
    "th_outliers_early":1000,
    "th_outliers":50, # value in pixels defining a point to be an outlier. If None, do not remove outliers.
    "output_path": "output/bundle_adjustment/"
}

def main(config=None,
         setup='setup.json',
         intrinsics='intrinsics.json',
         extrinsics='poses.json',
         landmarks='landmarks.json',
         filenames='filenames.json',
         iter1=40,
         iter2=80,
         dump_images=True):

    if config is not None:
        __config__ = utils.json_read(config)
    
    if iter1 is not None:
        __config__["max_nfev"] = iter1
    if iter2 is not None:
        __config__["max_nfev2"] = iter2    
    
    if dump_images:
        utils.mkdir(__config__["output_path"])

    setup = utils.json_read(setup)
    intrinsics = utils.json_read(intrinsics)
    extrinsics = utils.json_read(extrinsics)
    landmarks = utils.json_read(landmarks)
    filenames_images = utils.json_read(filenames)

    views = setup['views']
    print("-"*20)
    print("Views: {}".format(views))
    
    if __config__["each_training"]<2 or __config__["each_training"] is None:
        print("Use all the landmarks.")
    else:
        print("Subsampling the landmarks to 1 every {}.".format(__config__["each_training"]))    

    print("Preparing the input data...(this can take a while depending on the number of points to triangulate)")
    start = time.time()
    camera_params, points_3d, points_2d,\
    camera_indices, point_indices, \
    n_cameras, n_points, timestamps = build_input(views, intrinsics, extrinsics, 
                                                  landmarks, each=__config__["each_training"], 
                                                  view_limit_triang=4)
    print(time.time()-start)
    print("Sizes:")
    print("\t camera_params:", camera_params.shape)
    print("\t points_3d:", points_3d.shape)
    print("\t points_2d:", points_2d.shape)
    
    f0 = evaluate(camera_params, points_3d, points_2d,
                  camera_indices, point_indices,
                  n_cameras, n_points)    

    if dump_images:
        
        plt.figure()
        plt.plot(f0)
        plt.title("Residuals at initialization")
        plt.show()
        plt.savefig(os.path.join(__config__["output_path"], "initial_residuals.jpg"), bbox_inches='tight')
        
        
    if __config__["th_outliers_early"]==0 or __config__["th_outliers_early"] is None:
        print("No early outlier rejection.")
    else:
        print("Early Outlier rejection:")
        print("\t threshold outliers: {}".format(__config__["th_outliers_early"])) 
        
        f0_ = np.abs(f0.reshape(-1,2))
        mask_outliers = np.logical_or(f0_[:,0]>__config__["th_outliers_early"],f0_[:,1]>__config__["th_outliers_early"])
        point_indices = point_indices[~mask_outliers]
        camera_indices = camera_indices[~mask_outliers]
        points_2d = points_2d[~mask_outliers]
        optimized_points = np.int32(list(set(point_indices)))
        print("\t [Early rejection] Number of points considered outliers: ", sum(mask_outliers))

        if sum(mask_outliers)/len(mask_outliers)>0.5:
            print("!"*20)
            print("[Early rejection] More than half of the data points have been considered outliers! Something may have gone wrong.")
            print("!"*20)         
        
    if dump_images:
        
        f01 = evaluate(camera_params, points_3d, points_2d,
                      camera_indices, point_indices,
                      n_cameras, n_points)         
        
        plt.figure()
        plt.plot(f01)
        plt.title("Residuals after early outlier rejection")
        plt.show()
        plt.savefig(os.path.join(__config__["output_path"], "early_outlier_rejection_residuals.jpg"), bbox_inches='tight')        
        
    if __config__["bounds"]:
        print("Bounded optimization:")
        print("\t LB(x)=x-bound; UB(x)=x+bound")
        print("\t rvec bounds=({},{},{})".format(*__config__["bounds_cp"][:3]))
        print("\t tvec bounds=({},{},{})".format(*__config__["bounds_cp"][3:6]))
        print("\t k bounds=(fx={},fy={},c0={},c1={})".format(*__config__["bounds_cp"][6:10]))
        print("\t dist bounds=({},{},{},{},{})".format(*__config__["bounds_cp"][10:]))
        print("\t 3d points bounds=(x={},y={},z={})".format(*__config__["bounds_pt"]))
    else:
        print("Unbounded optimization.")
        
    print("Least-Squares optimization of the 3D points:")
    print("\t optimize camera parameters: {}".format(False))
    print("\t optimize 3d points: {}".format(True))
    print("\t ftol={:0.3e}".format(__config__["ftol"]))
    print("\t xtol={:0.3e}".format(__config__["xtol"]))
    print("\t loss={} f_scale={:0.2f}".format(__config__["loss"], __config__['f_scale']))
    print("\t max_nfev={}".format(__config__["max_nfev"]))
        
    points_3d_ref = bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, 
                                     point_indices, n_cameras, n_points, 
                                     optimize_camera_params=False, 
                                     optimize_points=True, 
                                     ftol=__config__["ftol"], xtol=__config__["xtol"],
                                     loss=__config__['loss'], f_scale=__config__['f_scale'],
                                     max_nfev=__config__["max_nfev"], 
                                     bounds=__config__["bounds"], 
                                     bounds_cp = __config__["bounds_cp"],
                                     bounds_pt = __config__["bounds_pt"], 
                                     verbose=True, eps=1e-12)        
        
    print("Least-Squares optimization of 3D points and camera parameters:")
    print("\t optimize camera parameters: {}".format(True))
    print("\t optimize 3d points: {}".format(True)) 
    print("\t ftol={:0.3e}".format(__config__["ftol"]))
    print("\t xtol={:0.3e}".format(__config__["xtol"]))
    print("\t loss={} f_scale={:0.2f}".format(__config__["loss"], __config__['f_scale']))
    print("\t max_nfev={}".format(__config__["max_nfev"]))    
        
    new_camera_params, new_points_3d = bundle_adjustment(camera_params, points_3d_ref, points_2d, camera_indices, 
                                                         point_indices, n_cameras, n_points, 
                                                         optimize_camera_params=__config__["optimize_camera_params"], 
                                                         optimize_points=__config__["optimize_points"], 
                                                         ftol=__config__["ftol"], xtol=__config__["xtol"],
                                                         loss=__config__['loss'], f_scale=__config__['f_scale'],
                                                         max_nfev=__config__["max_nfev"], 
                                                         bounds=__config__["bounds"], 
                                                         bounds_cp = __config__["bounds_cp"],
                                                         bounds_pt = __config__["bounds_pt"], 
                                                         verbose=True, eps=1e-12)

    
    f1 = evaluate(new_camera_params, new_points_3d, points_2d, 
                  camera_indices, point_indices, 
                  n_cameras, n_points)

    avg_abs_res = np.abs(f1).mean()
    print("Average absolute residual: {:0.2f} over {} points.".format(avg_abs_res, len(f1)/2))
    if avg_abs_res>50:
        print("!"*20)
        print("The average absolute residual error is higher than 50 pixels ({:0.2f})! Something may have gone wrong.".format(avg_abs_res))
        print("!"*20)
            
    if dump_images:
        plt.figure()
        plt.plot(f1)
        plt.title("Residuals after optimization")
        plt.show()
        plt.savefig(os.path.join(__config__["output_path"], "optimized_residuals.jpg"), bbox_inches='tight')        

    # Find ouliers points and remove them form the optimization.
    # These might be the result of inprecision in the annotations.
    # in this case we remove the resduals higher than 20 pixels.
    if __config__["th_outliers"]==0 or __config__["th_outliers"] is None:
        print("No outlier rejection.")
    else:
        print("Outlier rejection:")
        print("\t threshold outliers: {}".format(__config__["th_outliers"])) 
        print("\t max_nfev={}".format(__config__["max_nfev2"]))

        f1_ = np.abs(f1.reshape(-1,2))
        mask_outliers = np.logical_or(f1_[:,0]>__config__["th_outliers"],f1_[:,1]>__config__["th_outliers"])
        point_indices = point_indices[~mask_outliers]
        camera_indices = camera_indices[~mask_outliers]
        points_2d = points_2d[~mask_outliers]
        optimized_points = np.int32(list(set(point_indices)))
        print("\t Number of points considered outliers: ", sum(mask_outliers))
        
        if sum(mask_outliers)>0:
        
            if sum(mask_outliers)/len(mask_outliers)>0.5:
                print("!"*20)
                print("More than half of the data points have been considered outliers! Something may have gone wrong.")
                print("!"*20)            

            print("\t New sizes:")
            print("\t\t camera_params:", camera_params.shape)
            print("\t\t points_3d:", points_3d.shape)
            print("\t\t points_2d:", points_2d.shape)

            new_camera_params, new_points_3d = bundle_adjustment(camera_params, points_3d_ref, points_2d, camera_indices, 
                                                                 point_indices, n_cameras, n_points, 
                                                                 optimize_camera_params=__config__["optimize_camera_params"], 
                                                                 optimize_points=__config__["optimize_points"], 
                                                                 ftol=__config__["ftol"], xtol=__config__["xtol"],
                                                                 loss=__config__['loss'], f_scale=__config__['f_scale'],
                                                                 max_nfev=__config__["max_nfev2"], 
                                                                 bounds=__config__["bounds"], 
                                                                 bounds_cp = __config__["bounds_cp"],
                                                                 bounds_pt = __config__["bounds_pt"], 
                                                                 verbose=True, eps=1e-12)


            f2 = evaluate(new_camera_params, new_points_3d, points_2d, 
                          camera_indices, point_indices, 
                          n_cameras, n_points)

            avg_abs_res = np.abs(f2).mean()
            print("Average absolute residual: {:0.2f} over {} points.".format(avg_abs_res, len(f1)/2))
            if avg_abs_res>50:
                print("!"*20)
                print("The average absolute residual error (after outlier removal) is higher than 50 pixels ({:0.2f})! Something may have gone wrong.".format(avg_abs_res))
                print("!"*20)

            if dump_images:
                plt.figure()
                plt.plot(f2)
                plt.title("Residuals after outlier removal")
                plt.show()
                plt.savefig(os.path.join(__config__["output_path"], "optimized_residuals_outliers_removal.jpg"),
                            bbox_inches='tight')
        
    if __config__["each_visualisation"]<2 or __config__["each_visualisation"] is None:
        print("Visualise all the annotations.")
    else:
        print("Subsampling the annotations to visualise to 1 every {}.".format(__config__["each_visualisation"]))    
        
    path = __config__['output_path'] if dump_images else None
    visualisation(setup, landmarks, filenames_images, 
                  new_camera_params, new_points_3d, 
                  points_2d, camera_indices, each=__config__["each_visualisation"], path=path)

    ba_poses = {}
    for view, cp in zip(views, new_camera_params):
        K, R, t, dist = unpack_camera_params(cp)
        ba_poses[view] = {"R":R.tolist(), "t":t.tolist(), "K":K.tolist(), "dist":dist.tolist()}
    ba_points = {"points_3d": new_points_3d[optimized_points].tolist(), 
                 "timestamp":np.array(timestamps)[optimized_points].tolist()}  

    utils.json_write("ba_poses.json", ba_poses)
    utils.json_write("ba_points.json", ba_points)
    
if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()  
    parser.add_argument("--config", "-c", type=str, required=False, default=None,
                        help='JSON file containing the config. parameters for the bundle adjusment')    
    parser.add_argument("--setup", "-s", type=str, required=True, default="setup.json",
                        help='JSON file containing the camera setup')
    parser.add_argument("--intrinsics", "-i", type=str, required=True, default="intrinsics.json",
                        help='JSON file containing the intrinsic parameters for each view')
    parser.add_argument("--extrinsics", "-e", type=str, required=True, default="extrinsics.json",
                        help='JSON file containing the extrinsic parameters for each view')    
    parser.add_argument("--landmarks", "-l", type=str, required=True, default="landmarks.json",
                        help='JSON file containing the landmarks for each view')
    parser.add_argument("--filenames", "-f", type=str, required=False, default="filenames.json",
                        help='JSON file containing one filename of an image for each view. Used onyl if --dump_images is on')

    parser.add_argument("--dump_images", "-d", default=False, const=True, action='store_const',
                        help='Saves images for visualisation') 
    
    parser.add_argument("--iter1", "-it1", type=int, required=False, default=None,
                        help='Maximum number of iterations of the first optimization')
    parser.add_argument("--iter2", "-it2", type=int, required=False, default=None,
                        help='Maximum number of iterations of the second optimization after outlier rejection')     
    
    args = parser.parse_args()

    main(**vars(args))

# python bundle_adjustment.py -s setup.json -i intrinsics.json -e extrinsics.json -l landmarks.json -f filenmaes.json --dump_images 