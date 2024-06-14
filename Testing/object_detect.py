import numpy as np
import math
import pickle
import cv2
import time
import threading


global p0, p1, p2, p3, p4, p20, p21, p22, p23
count = 0
first = 1
marker_size_mm = 20
marker_size_m = 20
coordinates_3d_work = []
coordinates_3d_drop = []
move_arm = False


focal_length_px = 400

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
    marker_position = []
    marker_positio = {}
    global count

    if len(corners) > 0:
        ids = ids.flatten()
        i = 0
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

            cv2.putText(image, f"ID: {markerID} Dist: {distance:.2f}cm", (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            marker_position.append([cX, cY, distance, markerID])
            marker_positio[markerID] = (cX, cY)
            i += 1

    return image, marker_position, marker_positio

def pap_zones_display(marker_positions, img, key):
    global p0, p1, p2, p3, p4, p20, p21, p22, p23
    global first
    position_zero = None
    working_zone = {}
    dropping_zone = {}

    if key == ord("z") or first:
        first = 0

        p0 = marker_positions.get(0)
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
        working_zone = [p1, p2, p3, p4]

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
        dropping_zone = [p20, p21, p22, p23]

    if None in [p0]:
        cv2.putText(img, 'ZERO POSITION:', (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, 'NOT DETECTED', (175, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(img, 'ZERO POSITION:', (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, 'DETECTED', (175, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        position_zero = p0

    return img, working_zone, dropping_zone, position_zero

def is_within_zone(marker, working_zone):
    poly = np.array(working_zone, dtype=np.int32)
    if cv2.pointPolygonTest(poly, (marker[0], marker[1]), False) >= 0:
        return True
    else:
        return False

def calculate_3d_coordinates(marker_positions, working_zone, dropping_zone, position_zero, focal_length_px, marker_size_m, img):
    obj = 0
    coordinates_3d_work = []
    coordinates_3d_drop = []
    for marker_position in marker_positions:
        #print(marker_position)
        if 30 <= marker_position[3] < 50:
            p1, p2, p3, p4 = working_zone

            if None in [p1, p2, p3, p4]:
                return None

            x_w = marker_position[0]
            y_w = marker_position[1]
            z_w = marker_position[2]


            x_c = ((y_w - position_zero[1]) * z_w / focal_length_px) + 80
            y_c = -(x_w - position_zero[0]) * z_w / focal_length_px
            z_c = z_w

            if is_within_zone(marker_position, working_zone) and 30 <= marker_position[3] < 40:
                coordinates_3d_work.append([x_c, y_c, z_c, marker_position[3]])
                obj+=1
                cv2.putText(img, f"Obj_ID {marker_position[3]}: (X:{x_c:.4f} Y:{y_c:.4f} Z:{z_c:.4f})", (40, 130 + 20 * (obj)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if is_within_zone(marker_position, dropping_zone) and 40 <= marker_position[3] < 50:
                coordinates_3d_drop.append([x_c, y_c, z_c, marker_position[3]])
                obj+=1
                cv2.putText(img, f"Zone_ID {marker_position[3]}: (X:{x_c:.4f} Y:{y_c:.4f} Z:{z_c:.4f})", (40, 130 + 20 * (obj)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    return img, coordinates_3d_work, coordinates_3d_drop

def print_positions(coordinates_3d, detected_markers,detected_time_markers):
    current_time = time.time()

    for markerID in range(30, 39):
        if markerID in detected_markers:
            timestamp, printed = detected_time_markers[markerID]
            if current_time > timestamp + 5 and printed == 0:
                print(f"{coordinates_3d[markerID]}")
                print(f"{coordinates_3d[markerID + 10]}")
                detected_time_markers[markerID] = (timestamp, 1)

"""def move_objects(coordinates_3d, marker_positions, working_zone, dropping_zone):
    detected_time_markers = {}

    for markerID in range(30, 39):
        if markerID in coordinates_3d and is_within_zone(marker_positions[markerID], working_zone):
            
            if markerID + 10 in coordinates_3d and is_within_zone(marker_positions[markerID + 10], dropping_zone):
                if detected_time_markers[markerID]


    return
 """
def timer_callback():
    global coordinates_3d_work
    global coordinates_3d_drop
    found = 0
    if len(coordinates_3d_work) < timer_callback.count:
        timer_callback.count = len(coordinates_3d_work)
    if len(coordinates_3d_work) > timer_callback.count:
        timer_callback.count = len(coordinates_3d_work)
        timer_callback.delay = 50
    if timer_callback.delay == 0:
        for coord in coordinates_3d_drop:
            for coord_work in coordinates_3d_work:
                if coord_work[3]+10 in coord and not found:
                    print(f'{coord_work[0]} {coord_work[1]} 10\n')
                    print('200 0 200\n')
                    print(f'{coord[0]} {coord[1]} 50\n')
                    found = True
        timer_callback.delay = 30
    else:
        timer_callback.delay -= 1
    threading.Timer(0.1, timer_callback).start()

timer_callback.count = 0
timer_callback.delay = 60


    # main
aruco_type = "DICT_4X4_1000"

cameraMatrix, dist = pickle.load(open('calibration.pkl', 'rb'))
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (1280,720), 1, (1280,720))
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

last_printed_positions = {}
timer_callback()

while cap.isOpened():

    ret, img = cap.read()
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    x, y, m, n = roi
    img = dst[y:y + n, x:x + m]

    h, w, _ = img.shape

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    corners, ids, rejected = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    detected_markers, marker_positions, marker_positi = aruco_display(corners, ids, rejected, img)

    key = cv2.waitKey(1)

    img,  working_zone, dropping_zone, position_zero = pap_zones_display(marker_positi, img, key)

    if working_zone and position_zero:
        img, coordinates_3d_work, coordinates_3d_drop = calculate_3d_coordinates(marker_positions, working_zone, dropping_zone, position_zero, focal_length_px, marker_size_mm, img)

        #markers,detected_time_markers = move_objects(coordinates_3d, marker_positions, working_zone, dropping_zone)
        #print_positions(coordinates_3d, markers, detected_time_markers)

    cv2.imshow("zones", img)

    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
