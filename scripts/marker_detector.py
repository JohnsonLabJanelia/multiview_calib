import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, type=str)

args = parser.parse_args()

frame = cv2.imread(args.image)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
dictionary = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, _ = detector.detectMarkers(gray)

## draw image the corners, ids
if len(corners) > 0:
    # flatten the ArUco IDs list
    ids = ids.flatten()
    # loop over the detected ArUCo corners
    for (markerCorner, markerID) in zip(corners, ids):
        (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        cv2.line(frame, topLeft, topRight, (0, 255, 0), 15)
        cv2.line(frame, topRight, bottomRight, (0, 0, 255), 15)
        cv2.line(frame, bottomRight, bottomLeft, (0, 0, 255), 15)
        cv2.line(frame, bottomLeft, topLeft, (0, 0, 255), 15)
        # compute and draw the center (x, y)-coordinates of the ArUco marker                        
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(frame, (cX, cY), 15, (0, 0, 255), -1)                        
        # draw the ArUco marker ID on the frame
        cv2.putText(frame, str(markerID),
            (topLeft[0], topLeft[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            3, (0, 255, 0), 15)
        
    image_resize = cv2.resize(frame, (1604, 1100))
    cv2.imshow("{}".format(args.image), image_resize)
    key = cv2.waitKey(0)
    