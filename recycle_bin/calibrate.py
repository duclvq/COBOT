import cv2
import numpy as np
import glob
import os
CHESSBOARD_SIZE = (9, 7)  # (columns, rows) of internal corners
SQUARE_SIZE = 30   # Size of each square in mm (for real-world scaling)
CAMERA_NUMBER = 0  # Change to 1 if using an external camera
MATRIX_DIR = 'matrix/'
if not os.path.exists(MATRIX_DIR):
    os.makedirs(MATRIX_DIR)
def _capture_chessboard_images():
    """
    Capture images of a chessboard pattern using the webcam.
    This function captures images until the 'q' key is pressed and saves them in the 'capture' directory.
    """
    cap = cv2.VideoCapture(CAMERA_NUMBER)  # Use 0 for the default camera, or change to 1 for an external camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    # clear and create the capture directory
    import os
    if os.path.exists('capture'):
        for f in glob.glob('capture/*'):
            os.remove(f)
    else:
        os.makedirs('capture')
    # Create a window to display the camera feed
    cv2.namedWindow('Chessboard Capture')
    count = 0
    frame_number = 0
    C_pressed = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame_number += 1
        # add text to the frame, press C to start capturing
        if not C_pressed:
            cv2.putText(frame, f"*", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 5)
            
            cv2.putText(frame, "Press 'C' to start capturing images", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        else:
            cv2.putText(frame, f"Captured {count} images", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            if frame_number % 5 == 0:  # Display every 5th frame
                cv2.putText(frame, f"*", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

        cv2.imshow('Chessboard Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if not C_pressed:
                # print("Capturing images...")
                C_pressed = True
        if C_pressed:
            if frame_number % 10 == 0:  # Capture every 10th frame
                filename = f'capture/frame_{count:02d}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Captured {filename}")
                count += 1
        if key == ord('q'):
            print("Capture stopped by user.")
            break
    cap.release()
    cv2.destroyAllWindows()
    
def camera_calibration(CHESSBOARD_SIZE, SQUARE_SIZE):
    """
    Perform camera calibration using chessboard images.
    This function finds chessboard corners in the images, computes the camera matrix and distortion coefficients,
    and saves them for later use.
    """
    # Chessboard dimensions (change based on your printed pattern)
    

    # Prepare object points (3D points in real-world space)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale by square size

    # Lists to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Load chessboard images
    images = glob.glob('capture/*.jpg')  # Adjust path to your images

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display corners
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Save calibration results
    np.save(f"{MATRIX_DIR}camera_matrix.npy", camera_matrix)
    np.save(f"{MATRIX_DIR}dist_coeffs.npy", dist_coeffs)
    print("Camera Matrix (Intrinsic Parameters):\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    print("Calibration completed and results saved.")
    return camera_matrix, dist_coeffs
if __name__ == "__main__":
    # Capture chessboard images
    _capture_chessboard_images()

    # Perform camera calibration
    camera_matrix, dist_coeffs = camera_calibration(CHESSBOARD_SIZE, SQUARE_SIZE)

    # Load calibration parameters
    camera_matrix = np.load("camera_matrix.npy")
    dist_coeffs = np.load("dist_coeffs.npy")

    # Read a distorted image
    images = glob.glob('capture/*.jpg')  # Adjust path to your images
    img = cv2.imread(images[0])
    h, w = img.shape[:2]

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort image
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
