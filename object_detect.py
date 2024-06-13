import numpy as np
import math
import pickle
import cv2

global p1, p2, p3, p4, p20, p21, p22, p23
first = 1
marker_size_mm = 35
marker_size_m = 35

focal_length_px = 800

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

def calculate_distance(marker_size_m, focal_length_px, marker_pixel_width):
    return (marker_size_m * focal_length_px) / marker_pixel_width


def aruco_display(corners, ids, rejected, image):
    marker_position = {}
    marker_positio = {}

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

            topLeft_np = np.array(topLeft)
            topRight_np = np.array(topRight)
            marker_pixel_width = np.linalg.norm(topRight_np - topLeft_np)
            distance = calculate_distance(marker_size_m, focal_length_px, marker_pixel_width)

            cv2.putText(image, f"ID: {markerID} Dist: {distance:.2f}m", (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            marker_position[markerID] = (cX, cY, distance)
            marker_positio[markerID] = (cX, cY)

    return image, marker_position, marker_positio

def pap_zones_display(marker_positions, img, key):
    global p1, p2, p3, p4, p20, p21, p22, p23
    global first

    if key == ord("z") or first:
        first = 0

        p1 = marker_positions.get(1)
        p2 = marker_positions.get(2)
        p3 = marker_positions.get(3)
        p4 = marker_positions.get(4)

        p20 = marker_positions.get(20)
        p21 = marker_positions.get(21)
        p22 = marker_positions.get(22)
        p23 = marker_positions.get(23)

    # Cria uma cópia da imagem para o overlay
    overlay = img.copy()

    # Working zone
    if None in [p1, p2, p3, p4]:
        cv2.putText(img, f'P1: {p1}, P2: {p2}, P3: {p3}, P4: {p4}', (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, 'WORKING ZONE:', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, 'NOT DETECTED', (175, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        pts = np.array([p1[:2], p2[:2], p3[:2], p4[:2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0) )
        alpha = 0.2
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        cv2.putText(img, f'P1: {p1}, P2: {p2}, P3: {p3}, P4: {p4}', (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.putText(img, 'WORKING ZONE:', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, 'DETECTED', (175, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dropping zone
    if None in [p20, p21, p22, p23]:
        cv2.putText(img, f'P1: {p20}, P2: {p21}, P3: {p22}, P4: {p23}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.putText(img, 'DROPPING ZONE:', (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, 'NOT DETECTED', (175, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        pts = np.array([p20[:2], p21[:2], p22[:2], p23[:2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
        alpha = 0.2
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        cv2.putText(img, f'P1: {p20}, P2: {p21}, P3: {p22}, P4: {p23}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.putText(img, 'DROPPING ZONE:', (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, 'DETECTED', (185, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    working_zone = [p1, p2, p3, p4]
    dropping_zone = [p20, p21, p22, p23]

    return img, working_zone, dropping_zone


def calculate_3d_coordinates(marker_positions, working_zone, focal_length_px, marker_size_m, img):
    coordinates_3d = {}
    for markerID, marker_position in marker_positions.items():
        if markerID >= 30:
            p1, p2, p3, p4 = working_zone

            if None in [p1, p2, p3, p4]:
                return None

            x_w = marker_position[0]
            y_w = marker_position[1]
            z_w = marker_position[2]

            # Coordenadas 3D no espaço da câmera
            x_c = -(x_w - p1[0]) * z_w / focal_length_px
            y_c = -(y_w - p1[1]) * z_w / focal_length_px
            z_c = z_w

            coordinates_3d[markerID] = (x_c, y_c, z_c)
            cv2.putText(img, f"Obj_ID {markerID}: (X:{x_c:.4f} Y:{y_c:.4f} Z:{z_c:.4f})", (40, 110 + 20 * (markerID - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return img


#main
aruco_type = "DICT_4X4_1000"

cameraMatrix, dist = pickle.load(open('calibration.pkl', 'rb'))
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (1280,720), 1, (1280,720))
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)



# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():

    ret, img = cap.read()
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    x, y, m, n = roi
    img = dst[y:y + n, x:x + m]

    h, w, _ = img.shape

    width = 1500
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    corners, ids, rejected = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    detected_markers, marker_positions, marker_positi = aruco_display(corners, ids, rejected, img)

    key = cv2.waitKey(1)

    img,  working_zone, dropping_zone = pap_zones_display(marker_positi, img, key)

    # Calculate 3D coordinates for all markers with ID >= 30
    img = calculate_3d_coordinates(marker_positions, working_zone, focal_length_px, marker_size_m, img)

    cv2.imshow("zones", img)

    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
