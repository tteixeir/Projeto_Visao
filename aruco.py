import numpy as np
import math
import pickle
import cv2
import sys
import time

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def side_distance(corners):
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    sides = {
	"left" : math.sqrt(int(bottomLeft[0]-topLeft[0])**2 + int(bottomLeft[1]-topLeft[1])**2),
    "top" : math.sqrt(int(topRight[0]-topLeft[0])**2 + int(topRight[1]-topLeft[1])**2),
    "right" : math.sqrt(int(topRight[0]-bottomRight[0])**2 + int(topRight[1]-bottomRight[1])**2),
    "bottom" : math.sqrt(int(bottomLeft[0]-bottomRight[0])**2 + int(bottomLeft[1]-bottomRight[1])**2)
	}
    return sides

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shBeId = np.dot(Rt,R)
    I = np.identity(3, dtype= R.dtype)
    n = np.linalg.norm(I - shBeId)
    return n < 1e-6

def rotationMatrixEulerAngles(R):
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,1]*R[1,1])
    
    if not (sy < 1e-6):
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x,y,z])

def aruco_display(corners, ids, rejected, image):
    
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
   
			#sides = side_distance(corners)
   
			#print("{:.2f}".format(sides["left"])+",{:.2f}".format(sides["top"])+",{:.2f}".format(sides["right"])+",{:.2f}".format(sides["bottom"]))
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_PLAIN,
				0.5, (0, 255, 0), 2)
			#print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image



def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(cv2.aruco_dict,parameters)


    corners, ids, rejected_img_points = detector.detectMarkers(gray)

        
    if len(corners) > 0:
        for i in range(0, len(ids)):
           
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 22, matrix_coefficients,
                                                                       distortion_coefficients)
            rvec = rvec[0][0]
            tvec = tvec[0][0]
            
            rvec_flip = rvec * -1
            tvec_flip = tvec * -1
            rot_mat, jac = cv2.Rodrigues(rvec_flip)
            realworld_tvec = np.dot(rot_mat,tvec_flip)
            
            pitch, roll, yaw = rotationMatrixEulerAngles(rot_mat)
            
            if (ids[i] == 11):
                frame = aruco_display(corners[i], ids[i], rejected_img_points ,frame)
                t_vec_str = "x = %4.0fmm y = %4.0fmm direction = %4.0f degrees"%(realworld_tvec[0],realworld_tvec[1],math.degrees(yaw))
                
                cv2.putText(frame, t_vec_str, (20, 400), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),2)
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 50)
             
    return frame

aruco_type = "DICT_4X4_50"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters()

cameraMatrix, dist = pickle.load(open('calibration.pkl', 'rb'))

cap = cv2.VideoCapture(1)
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (1280,720), 1, (1280,720))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



while cap.isOpened():
    
    ret, img = cap.read()
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    
    x, y, w, h = roi
    img = dst[y:y+h, x:x+w]
    
    output = pose_estimation(img, ARUCO_DICT[aruco_type], cameraMatrix, dist)

    cv2.imshow('Estimated Pose', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()