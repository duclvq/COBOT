import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import time
from recycle_bin.depth_estimatation import DepthEstimation  
# Initialize MediaPipe Hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Initialize depth estimation model
depth_estimator = DepthEstimation()
# Open video capture
CAMERA_NUMBER = 0  # Change to 1 if using an external camera
MATRIX_DIR = 'matrix/'
CAPTURE_DIR = 'capture/'
CHESSBOARD_SIZE = (9, 7)
SQUARE_SIZE = 30

# Ensure the matrix directory exists
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)
if not os.path.exists(MATRIX_DIR):
    os.makedirs(MATRIX_DIR)
cap = cv2.VideoCapture(CAMERA_NUMBER)

# Load camera calibration parameters
camera_matrix = np.load(f"{MATRIX_DIR}camera_matrix.npy")
dist_coeffs = np.load(f"{MATRIX_DIR}dist_coeffs.npy")



# Load rotation and translation vectors
if not os.path.exists(f"{MATRIX_DIR}rvec.npy") or not os.path.exists(f"{MATRIX_DIR}tvec.npy"):
    print("Error: Rotation and translation vectors not found. Please run calibrate.py first.")
    exit()
rvec = np.load(f"{MATRIX_DIR}rvec.npy")
tvec = np.load(f"{MATRIX_DIR}tvec.npy")

# Get video properties
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))



# chessboard_depth = tvec[2]
# def estimate_chessboard_distance(image, pattern_size, square_size, camera_matrix, dist_coeffs):
#     """
#     Estimates the distance from the camera to a chessboard.

#     Args:
#         image: The input image (grayscale or color).
#         pattern_size: Tuple (columns, rows) of internal corners of the chessboard.
#         square_size: The size of a chessboard square in real-world units (e.g., meters or mm).
#         camera_matrix: The camera intrinsic matrix.
#         dist_coeffs: The camera distortion coefficients.

#     Returns:
#         The estimated distance to the chessboard in the same units as square_size,
#         or None if the chessboard is not found.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

#     if ret:
#         # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#         objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
#         objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
#         objp = objp * square_size

#         # Find the rotation and translation vectors.
#         ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
#         if ret:
#             # The distance is the Z component of the translation vector
#             distance = tvec[2][0]
#             return distance
#     return None
def image_to_chessboard_3D_coords(image_point, depth, camera_matrix, dist_coeffs, rvec, tvec):
    """
    Convert a 2D image point with depth into a 3D point in the chessboard's coordinate system.
    :param image_point: (u, v) pixel coordinates in the image
    :param depth: Z value (real-world depth in mm or meters)
    :param camera_matrix: Intrinsic camera matrix (3x3)
    :param dist_coeffs: Distortion coefficients
    :param rvec: Rotation vector from solvePnP
    :param tvec: Translation vector from solvePnP
    :return: 3D point (X, Y, Z) in the chessboard coordinate system
    """
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Convert image point to normalized camera coordinates
    u, v = image_point
    uv1 = np.array([[u, v, 1]], dtype=np.float32).T  # (3,1)
    
    # Invert camera matrix
    K_inv = np.linalg.inv(camera_matrix)
    
    # Get camera frame coordinates (before applying depth)
    X_cam = K_inv @ uv1 * depth  # Scale by depth
    
    # Convert to world coordinates (chessboard)
    X_world = np.linalg.inv(R) @ (X_cam - tvec)
    
    return X_world.flatten()



def get_chessboard_root_2D(rvec, tvec, camera_matrix, dist_coeffs):
    """
    Projects the chessboard's origin (0,0,0 in its world coordinate system)
    to 2D image coordinates.

    Args:
        rvec: Rotation vector from solvePnP.
        tvec: Translation vector from solvePnP.
        camera_matrix: Intrinsic camera matrix.
        dist_coeffs: Distortion coefficients.

    Returns:
        Tuple (u, v) of pixel coordinates in the image.
    """
    origin_3d = np.array([[0, 0, 0]], dtype=np.float32)
    projected_points, _ = cv2.projectPoints(origin_3d, rvec, tvec, camera_matrix, dist_coeffs)
    u = int(projected_points[0][0][0])
    v = int(projected_points[0][0][1])
    return (u, v)

K =  np.load(f"{MATRIX_DIR}camera_matrix.npy")


# capture the first frame
ret, frame = cap.read()
# cv2.imshow('frame', frame)
# cv2.waitKey(0)
# get the first coorners
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
depth_color = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
depth = np.zeros((frame_height, frame_width), dtype=np.uint8)


# Prepare object points (3D points in real-world space)
objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Create axis points
axis = np.float32([[3*SQUARE_SIZE, 0, 0], [0, 3*SQUARE_SIZE, 0], [0, 0, -3*SQUARE_SIZE]]).reshape(-1, 3)
imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

# distance_to_board = estimate_chessboard_distance(frame, CHESSBOARD_SIZE, SQUARE_SIZE, camera_matrix, dist_coeffs)
distance_to_board = tvec[2][0]  # Z component of the translation vector
if ret:
    print("Rotation Vector:\n", rvec)
    print("Translation Vector:\n", tvec)
    print("Distance to chessboard (Z): ", distance_to_board)
if distance_to_board is not None:
    print(f"Estimated distance to chessboard: {distance_to_board:.2f} mm")
wrist_depth = 0
new_x = 0
new_y = 0
# Shared variables for threading
depth_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
depth = np.zeros((frame_height, frame_width), dtype=np.uint8)
depth_color = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
depth_lock = threading.Lock()
root_coor_2D = get_chessboard_root_2D(rvec, tvec, camera_matrix, dist_coeffs)
max_value = 255
from PIL import Image
def run_depth_in_thread():
    global depth_frame
    global depth_color
    global depth 
    global max_value  
    while True:
        with depth_lock:
            if depth_frame is not None:
                # convert depth_frame to PIL Image
                depth_frame_pil = Image.fromarray(cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB))
                print("depth_frame_pil size: ", depth_frame_pil.size)
                depth_map = depth_estimator.estimate_depth(depth_frame_pil)
                # get value at the root of the chessboard
                if depth_map is None:
                    print("Depth estimation failed.")
                    continue
                np_depth_map = np.array(depth_map)
                depth = np_depth_map 
                max_value = 255 - np_depth_map[root_coor_2D[1], root_coor_2D[0]]
                print("Max value at root: ", max_value)
                # print("Depth map shape: ", depth_map.shape)
                # # print min and max depth values
                print("Min depth: ", np.min(np_depth_map), " Max depth: ", np.max(np_depth_map))
                # Normalize the depth map to 0-255 for visualization
                # depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
                # depth = depth.cpu().numpy().astype(np.uint8)
                depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                # save the depth map
                cv2.imwrite("depth_map.png", depth)
                cv2.imwrite("depth_color.png", depth_color)



# Start the depth estimation thread
depth_thread = threading.Thread(target=run_depth_in_thread)

depth_thread.daemon = True
depth_thread.start()
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Convert the image from BGR to RGB (MediaPipe requirement)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and detect hands
        results = hands.process(image)
        if frame_count % 10 == 0:
            with depth_lock:
                depth_frame = image.copy() 
            # print("Time taken: ", time.time() - depth_start_time)
        wrist_y = 0
        wrist_x = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract wrist landmark (Landmark 0)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, _ = image.shape
                
                # Convert wrist coordinates from normalized (0 to 1) to pixel values
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                ori_wrist_z = wrist.z * 1000  # Approximate Z in mm (MediaPipe gives a negative Z)
                wrist_z = 747  
                # Convert wrist 2D image point to 3D world coordinates
                image_points = np.array([[wrist_x, wrist_y]], dtype=np.float32)

                # Undistort the image points
                undistorted_points = cv2.undistortPoints(image_points, camera_matrix, dist_coeffs)
                fx = K[0, 0]
                fy = K[1, 1]
                c_x = K[0, 2]
                c_y = K[1, 2]
                new_x = undistorted_points[0][0][0] * fx + c_x
                new_y = undistorted_points[0][0][1] * fy + c_y
                
                print(new_x, new_y)
                ori_wrist_z = wrist.z * 1000000   # Approximate Z in mm (MediaPipe gives a negative Z)
                cv2.putText(image, f"Z_mediapipe: {ori_wrist_z:.2f} x1M ", (wrist_x, wrist_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0 , 0, 255), 2)
                
                
                # Get the depth at the wrist location
                wrist_depth = depth[int(wrist_y), int(wrist_x)]
                print("wrist depth: ", wrist_depth)
                wrist_depth = 255 - wrist_depth
                wrist_depth = float(wrist_depth) / max_value * distance_to_board
                wrist_cobot = image_to_chessboard_3D_coords(image_points[0], wrist_depth, camera_matrix, dist_coeffs, rvec, tvec)

                # # Draw information on the image
                cv2.putText(image, f"X: {wrist_cobot[0]:.2f} mm", (wrist_x, wrist_y - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Y: {wrist_cobot[1]:.2f} mm", (wrist_x, wrist_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, f"Z: {wrist_cobot[2]:.2f} mm", (wrist_x, wrist_y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw a circle at the wrist location
                cv2.circle(image, (wrist_x, wrist_y), 10, (0, 255, 0), -1)
                image = cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 50)
                # draw the distance to the chessboard
                cv2.putText(image, f"Distance Cam->chessboard: {distance_to_board:.2f} mm", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # distance to the wrist in the cobot frame
                wrist_cobot_distance = np.linalg.norm(wrist_cobot[:3])
                cv2.putText(image, f"Distance Cam->wrist: {wrist_depth:.2f} mm", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(image, f"Distance wrist->chessboard: {wrist_cobot_distance:.2f} mm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        # Display the image with hand tracking
        _depth_color = cv2.resize(depth_color, (int(frame_width/3), int(frame_height/3)))
        # depth at wrist location
        
        # cv2.putText(image, f"Z (camera): {depth_wrist:.2f} mm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(_depth_color, f"Z (camera->hand): {wrist_depth:.2f} mm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # put the wrist point on the depth image
        cv2.circle(_depth_color, (int(wrist_x/3), int(wrist_y/3)), 10, (0, 255, 0), -1)
        cv2.circle(_depth_color, (int(new_x/3), int(new_y/3)), 10, (0, 255, 0), -1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Tracking', image)
        cv2.imshow('Depth', _depth_color)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

# Release the camera and close OpenCV windows
cap.release()


cv2.destroyAllWindows()

