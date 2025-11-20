import cv2
import numpy as np
import glob
import os
matrix_dir = 'matrix/'
if not os.path.exists(matrix_dir):
    os.makedirs(matrix_dir)
# Chessboard parameters
chessboard_size = (9, 7)  # Number of inner corners per row and column
square_size = 30  # The size of a square in mm (or any unit)

# Prepare object points (3D points in real-world space)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Load camera intrinsic parameters (assumed to be pre-calibrated)
camera_matrix = np.load(f"{matrix_dir}camera_matrix.npy")
dist_coeffs = np.load(f"{matrix_dir}dist_coeffs.npy")

cap = cv2.VideoCapture(0)  # Change to your camera number or video file path
# Read a single frame to get the image path
ret, frame = cap.read()
cap.release()
if not ret:
    print("Failed to capture image from camera.")
    exit()
# Read image
image = frame
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect chessboard corners
found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

if found:
    # print(objp[:10])
    # print(corners[:10])
    # Find pose of the chessboard
    ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", tvec)
    
    # Extrinsic matrix (R|t)
    extrinsic_matrix = np.hstack((R, tvec))
    print("Extrinsic Matrix:\n", extrinsic_matrix)

    # Draw axis for visualization
    axis = np.float32([[3*square_size, 0, 0], [0, 3*square_size, 0], [0, 0, -3*square_size]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    image = cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 50)
    cv2.imshow("Pose Estimation", image)
    cv2.waitKey(0)

else:
    print("Chessboard not found!")

cv2.destroyAllWindows()

# save rotation matrix
np.save(os.path.join(matrix_dir, "R.npy"), R)
# save translation vector
np.save(os.path.join(matrix_dir, "T.npy"), tvec)
# save extrinsic matrix
np.save(os.path.join(matrix_dir, "extrinsic_matrix.npy"), extrinsic_matrix)
# save the rotation vector
np.save(os.path.join(matrix_dir, "rvec.npy"), rvec)
# save the translation vector
np.save(os.path.join(matrix_dir, "tvec.npy"), tvec)
# save the extrinsic matrix
np.save(os.path.join(matrix_dir, "extrinsic_matrix.npy"), extrinsic_matrix)
# save T_base_camera
T_base_camera = np.eye(4)
T_base_camera[:3, :3] = R
T_base_camera[:3, 3] = tvec.flatten()
np.save(os.path.join(matrix_dir, "T_base_camera.npy"), T_base_camera)

