import cv2
import argparse
import colorsys

def generate_distinct_colors(n):
    """Generate `n` distinct RGB colors evenly spaced in HSV space."""
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 1, 1)) for i in range(n)
    ]


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, type=str)
parser.add_argument("--width", required=True, type=int)
parser.add_argument("--height", required=True, type=int)
parser.add_argument("-sl", "--square_length", required=True, type=float)
parser.add_argument("-ml", "--marker_length", required=True, type=float)
parser.add_argument("-d","--dict", required=True, type=int)

"""
    "dictionary",int:
        dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,  DICT_4X4_1000=3,
        DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, DICT_6X6_50=8,
        DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, DICT_7X7_100=13,
        DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16
"""

args = parser.parse_args()

frame = cv2.imread(args.image)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


image_resize = cv2.resize(gray, (1604, 1100))
cv2.imshow("gray", image_resize)
key = cv2.waitKey(0)

threshold_value = 95
max_value = 255
_, img_thresh = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TOZERO)
image_resize_thresh = cv2.resize(img_thresh, (1604, 1100))
cv2.imshow("thresh", image_resize_thresh)
key = cv2.waitKey(0)

dictionary = args.dict
aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
num_points_thres = 6
board_size = (args.width, args.height)
board = cv2.aruco.CharucoBoard(board_size, args.square_length, args.marker_length, aruco_dict)
charuco_detector = cv2.aruco.CharucoDetector(board)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.00001)
number_of_markers = board_size[0] * board_size[1]
marker_colors = generate_distinct_colors(number_of_markers)

frame_detect = gray.copy()
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame_detect, aruco_dict)
print("Length of markers detected {}".format(len(corners)))

if (len(corners) >= num_points_thres):
    # flatten the ArUco IDs list
    ids = ids.flatten()
    # loop over the detected ArUCo corners
    image_copy = frame.copy()
    for (markerCorner, markerID) in zip(corners, ids):
        (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        cv2.line(image_copy, topLeft, topRight, (0, 255, 0), 15)
        cv2.line(image_copy, topRight, bottomRight, (0, 0, 255), 15)
        cv2.line(image_copy, bottomRight, bottomLeft, (0, 0, 255), 15)
        cv2.line(image_copy, bottomLeft, topLeft, (0, 0, 255), 15)
        # compute and draw the center (x, y)-coordinates of the ArUco marker                        
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image_copy, (cX, cY), 15, (0, 0, 255), -1)                        
        # draw the ArUco marker ID on the frame
        cv2.putText(image_copy, str(markerID),
            (topLeft[0], topLeft[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            3, (0, 255, 0), 15)
        
    image_resize = cv2.resize(image_copy, (1604, 1100))
    cv2.imshow("aruco", image_resize)
    key = cv2.waitKey(0)

    #charuco_detection = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = (charuco_detector.detectBoard(frame_detect))   
    if charuco_corners is not None and charuco_ids is not None:
        print("Detection: length of charuco_ids {}".format(len(charuco_ids)))
        obj_points, img_points = board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        print(obj_points.shape)
        print(obj_points)

        # SUB PIXEL DETECTION
        for corner in corners:
            cv2.cornerSubPix(
                gray,
                corner,
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=criteria,
            )

        res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if (
            res2[1] is not None
            and res2[2] is not None
            and len(res2[1]) >= num_points_thres
        ):
            image_copy = frame.copy()

            for pts_idx in range(res2[1].shape[0]):
                # breakpoint()
                cv2.circle(
                    image_copy,
                    (
                        int(res2[1][pts_idx, 0, 0]),
                        int(res2[1][pts_idx, 0, 1]),
                    ),
                    15,
                    marker_colors[pts_idx],
                    -1,
                )

                cv2.putText(
                    image_copy,
                    str(int(res2[2][pts_idx][0])),
                    (
                        int(res2[1][pts_idx, 0, 0]),
                        int(res2[1][pts_idx, 0, 1]),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    marker_colors[pts_idx],
                    5,
                )

            image_resize = cv2.resize(image_copy, (1604, 1100))
            cv2.imshow("board", image_resize)
            key = cv2.waitKey(0)
