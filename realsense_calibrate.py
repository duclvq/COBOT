import pyrealsense2 as rs
import numpy as np
import cv2
import os

CHESSBOARD_SIZE = (9, 7)  # (columns, rows) of internal corners
SQUARE_SIZE = 30  # Size of each square in mm
MATRIX_DIR = 'matrix/'
CAPTURE_DIR = 'capture/'

# Ensure directories exist
if not os.path.exists(MATRIX_DIR):
    os.makedirs(MATRIX_DIR)
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)


def capture_chessboard_images():
    """
    Capture chessboard images using RealSense camera for calibration.
    Press 's' to save image, 'q' to quit.
    """
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    print("Starting RealSense camera...")
    profile = pipeline.start(config)
    
    # Configure depth sensor for better quality
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
        depth_sensor.set_option(rs.option.laser_power, 360)
        print("✓ Depth sensor configured for high accuracy")
    except:
        print("⚠ Could not configure depth sensor")
    
    print("Camera started! Place chessboard in view.")
    print("\nControls:")
    print("  's' - Save current frame")
    print("  'q' - Quit and proceed to calibration")
    
    saved_count = 0
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Try to detect chessboard
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            
            # Draw chessboard if found
            display_image = color_image.copy()
            if ret:
                cv2.drawChessboardCorners(display_image, CHESSBOARD_SIZE, corners, ret)
                cv2.putText(display_image, "Chessboard detected!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "Chessboard not found", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(display_image, f"Saved: {saved_count} images", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('RealSense Calibration - Press S to save, Q to quit', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and ret:
                # Save image
                filename = f'{CAPTURE_DIR}calib_{saved_count:03d}.png'
                cv2.imwrite(filename, color_image)
                print(f"✓ Saved: {filename}")
                saved_count += 1
            elif key == ord('s') and not ret:
                print("✗ Cannot save - chessboard not detected!")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    return saved_count


def calibrate_camera():
    """
    Calibrate camera using captured chessboard images.
    """
    print("\n" + "="*50)
    print("Starting camera calibration...")
    print("="*50)
    
    # Prepare object points
    objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Find all calibration images
    import glob
    images = glob.glob(f'{CAPTURE_DIR}calib_*.png')
    
    if len(images) == 0:
        print("✗ No calibration images found!")
        return False
    
    print(f"Found {len(images)} calibration images")
    
    img_shape = None
    successful_images = 0
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            successful_images += 1
            print(f"  ✓ Image {idx + 1}/{len(images)}: Corners found")
        else:
            print(f"  ✗ Image {idx + 1}/{len(images)}: Corners not found")
    
    if successful_images < 3:
        print(f"\n✗ Not enough valid images ({successful_images}). Need at least 3!")
        return False
    
    print(f"\nCalibrating with {successful_images} images...")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None)
    
    if ret:
        # Save calibration parameters
        np.save(f'{MATRIX_DIR}camera_matrix.npy', camera_matrix)
        np.save(f'{MATRIX_DIR}dist_coeffs.npy', dist_coeffs)
        
        print("\n" + "="*50)
        print("✓ Calibration successful!")
        print("="*50)
        print("\nCamera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs.ravel())
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                              camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        print(f"\nMean reprojection error: {mean_error / len(objpoints):.4f} pixels")
        print(f"\nCalibration files saved to {MATRIX_DIR}")
        
        return True
    else:
        print("\n✗ Calibration failed!")
        return False


def capture_reference_frame():
    """
    Capture a reference frame with chessboard to get extrinsic parameters.
    """
    print("\n" + "="*50)
    print("Capturing reference frame for extrinsic parameters...")
    print("="*50)
    print("Position the chessboard as your coordinate system origin.")
    print("Press 'c' to capture when ready.")
    
    # Load camera matrix and distortion coefficients
    try:
        camera_matrix = np.load(f'{MATRIX_DIR}camera_matrix.npy')
        dist_coeffs = np.load(f'{MATRIX_DIR}dist_coeffs.npy')
    except:
        print("✗ Camera calibration not found! Run calibration first.")
        return False
    
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    profile = pipeline.start(config)
    
    # Configure depth sensor
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)
        depth_sensor.set_option(rs.option.laser_power, 360)
    except:
        pass
    
    # Prepare object points
    objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            
            display_image = color_image.copy()
            if ret:
                cv2.drawChessboardCorners(display_image, CHESSBOARD_SIZE, corners, ret)
                cv2.putText(display_image, "Chessboard detected! Press 'c' to capture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "Position chessboard in view", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Reference Frame Capture', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Solve PnP to get extrinsic parameters
                ret, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)
                
                if ret:
                    # Save extrinsic parameters
                    np.save(f'{MATRIX_DIR}rvec.npy', rvec)
                    np.save(f'{MATRIX_DIR}tvec.npy', tvec)
                    
                    # Convert rotation vector to matrix
                    R, _ = cv2.Rodrigues(rvec)
                    np.save(f'{MATRIX_DIR}R.npy', R)
                    
                    # Create transformation matrix (camera to chessboard)
                    T_chess_to_cam = np.eye(4)
                    T_chess_to_cam[:3, :3] = R
                    T_chess_to_cam[:3, 3] = tvec.flatten()
                    
                    # Invert to get camera to chessboard transform
                    T_cam_to_chess = np.linalg.inv(T_chess_to_cam)
                    np.save(f'{MATRIX_DIR}T_cam_to_chess.npy', T_cam_to_chess)
                    np.save(f'{MATRIX_DIR}extrinsic_matrix.npy', T_chess_to_cam)
                    
                    print("\n✓ Reference frame captured!")
                    print(f"Rotation vector:\n{rvec.flatten()}")
                    print(f"\nTranslation vector (mm):\n{tvec.flatten()}")
                    print(f"\nTransformation matrix saved to {MATRIX_DIR}")
                    
                    # Save reference image
                    cv2.imwrite(f'{CAPTURE_DIR}reference_frame.png', color_image)
                    print(f"Reference image saved to {CAPTURE_DIR}reference_frame.png")
                    
                    break
                else:
                    print("✗ Could not solve PnP!")
            elif key == ord('q'):
                print("Cancelled.")
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    return ret


def main():
    """
    Main calibration workflow for RealSense camera.
    """
    print("="*60)
    print("RealSense Camera Calibration for Hand-Eye Coordination")
    print("="*60)
    
    print("\nThis calibration process consists of 3 steps:")
    print("1. Capture chessboard images from different angles")
    print("2. Calculate camera intrinsic parameters")
    print("3. Capture reference frame for extrinsic parameters")
    
    input("\nPress Enter to start...")
    
    # Step 1: Capture calibration images
    print("\n--- STEP 1: Capture Calibration Images ---")
    print("Move the chessboard to different positions and angles.")
    print("Aim for at least 10-15 good images.")
    
    num_images = capture_chessboard_images()
    
    if num_images < 3:
        print(f"\n✗ Not enough images captured ({num_images}). Need at least 3!")
        print("Please run the script again and capture more images.")
        return
    
    # Step 2: Calibrate camera
    print("\n--- STEP 2: Camera Calibration ---")
    success = calibrate_camera()
    
    if not success:
        print("\n✗ Calibration failed!")
        return
    
    # Step 3: Capture reference frame
    print("\n--- STEP 3: Reference Frame Capture ---")
    success = capture_reference_frame()
    
    if success:
        print("\n" + "="*60)
        print("✓ CALIBRATION COMPLETE!")
        print("="*60)
        print("\nCalibration files created:")
        print(f"  - {MATRIX_DIR}camera_matrix.npy")
        print(f"  - {MATRIX_DIR}dist_coeffs.npy")
        print(f"  - {MATRIX_DIR}rvec.npy")
        print(f"  - {MATRIX_DIR}tvec.npy")
        print(f"  - {MATRIX_DIR}R.npy")
        print(f"  - {MATRIX_DIR}T_cam_to_chess.npy")
        print(f"  - {MATRIX_DIR}extrinsic_matrix.npy")
        print("\nYou can now run realsense_reader.py to track hand position!")
    else:
        print("\n✗ Reference frame capture failed!")


if __name__ == "__main__":
    main()
