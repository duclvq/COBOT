import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import os
from recycle_bin.realsense_hand_transform import HandCoordinateTransform

# Chessboard configuration
CHESSBOARD_SIZE = (9, 7)  # (columns, rows) of internal corners
SQUARE_SIZE = 30  # Size of each square in mm
MATRIX_DIR = 'matrix/'


class RealSenseReader:
    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize RealSense camera reader
        
        Args:
            width: Image width (default: 640)
            height: Image height (default: 480)
            fps: Frame rate (default: 30)
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # Create pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Create align object to align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get depth sensor and set depth units
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        print(f"RealSense camera initialized: {width}x{height} @ {fps}fps")
        print(f"Depth scale: {self.depth_scale}")
    
    def smooth_depth(self, depth_image, kernel_size=5):
        """
        Làm mịn depth image bằng Gaussian blur
        
        Args:
            depth_image: Depth image array
            kernel_size: Kernel size for smoothing (must be odd)
        
        Returns:
            Smoothed depth image
        """
        # Apply Gaussian blur to smooth depth
        smoothed = cv2.GaussianBlur(depth_image, (kernel_size, kernel_size), 0)
        return smoothed
    
    def fill_zero_depth(self, depth_image, iterations=2):
        """
        Lấp đầy các vùng depth = 0 bằng trung bình của các pixel lân cận
        
        Args:
            depth_image: Depth image array
            iterations: Số lần lặp để lấp đầy (mỗi lần lấp thêm 1 lớp)
        
        Returns:
            Depth image đã được lấp đầy
        """
        result = depth_image.copy()
        
        for _ in range(iterations):
            # Tìm các vùng bằng 0
            zero_mask = (result == 0)
            
            if not np.any(zero_mask):
                break
            
            # Tạo kernel 3x3 để tính trung bình lân cận
            kernel = np.ones((3, 3), dtype=np.float32)
            kernel[1, 1] = 0  # Không tính điểm trung tâm
            
            # Đếm số lượng pixel khác 0 xung quanh mỗi pixel
            count = cv2.filter2D((result > 0).astype(np.float32), -1, kernel)
            
            # Tính tổng giá trị xung quanh mỗi pixel
            sum_neighbors = cv2.filter2D(result, -1, kernel)
            
            # Tính trung bình (tránh chia cho 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                average = np.where(count > 0, sum_neighbors / count, 0)
            
            # Chỉ điền vào các vùng bằng 0
            result = np.where(zero_mask, average, result)
        
        return result
    
    def read(self):
        """
        Read RGB and depth images from camera
        
        Returns:
            rgb_image: RGB color image (numpy array)
            depth_image: Raw depth image (numpy array)
            depth_image_mm: Smoothed depth image in millimeters (numpy array)
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
        
        # Convert to numpy arrays
        rgb_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Convert depth to millimeters
        depth_image_mm = depth_image.astype(np.float32) * self.depth_scale * 1000
        
        # Bước 1: Lấp đầy các vùng depth = 0 bằng trung bình lân cận
        depth_image_mm = self.fill_zero_depth(depth_image_mm, iterations=3)
        
        # Bước 2: Làm mịn depth bằng Gaussian blur
        depth_image_mm = self.smooth_depth(depth_image_mm, kernel_size=5)
        
        return rgb_image, depth_image, depth_image_mm
    
    def get_intrinsics(self):
        """
        Get camera intrinsic parameters
        
        Returns:
            color_intrinsics: Color camera intrinsics
            depth_intrinsics: Depth camera intrinsics
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        return color_intrinsics, depth_intrinsics
    
    def stop(self):
        """Stop the camera pipeline"""
        self.pipeline.stop()
        print("RealSense camera stopped")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


def get_chessboard_origin_2D(rvec, tvec, camera_matrix, dist_coeffs, chessboard_size, square_size):
    """
    Get the bottom-left corner (origin) of the chessboard in 2D image coordinates.
    
    Args:
        rvec: Rotation vector
        tvec: Translation vector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        chessboard_size: (columns, rows) of internal corners
        square_size: Size of each square in mm
    
    Returns:
        Tuple (u, v) of pixel coordinates for the bottom-left origin
    """
    # The origin is at the bottom-left corner of the chessboard
    # In the chessboard coordinate system, this is at (0, rows*square_size, 0)
    origin_3d = np.array([[0, chessboard_size[1] * square_size, 0]], dtype=np.float32)
    projected_points, _ = cv2.projectPoints(origin_3d, rvec, tvec, camera_matrix, dist_coeffs)
    u = int(projected_points[0][0][0])
    v = int(projected_points[0][0][1])
    return (u, v)


def main():
    """Example usage of RealSenseReader with hand tracking"""
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    print("Starting RealSense camera with hand tracking...")
    print("Make sure RealSense camera is connected!")
    
    # Load camera calibration parameters and coordinate transformer
    try:
        camera_matrix = np.load(f"{MATRIX_DIR}camera_matrix.npy")
        dist_coeffs = np.load(f"{MATRIX_DIR}dist_coeffs.npy")
        rvec = np.load(f"{MATRIX_DIR}rvec.npy")
        tvec = np.load(f"{MATRIX_DIR}tvec.npy")
        
        # Initialize coordinate transformer
        transformer = HandCoordinateTransform(MATRIX_DIR)
        
        print("✓ Loaded calibration data from matrix/")
        calibration_loaded = True
    except Exception as e:
        print(f"Warning: Could not load calibration data: {e}")
        print("Chessboard origin and coordinates will not be displayed")
        print("Run realsense_calibrate.py first to calibrate the camera")
        calibration_loaded = False
        transformer = None
    
    try:
        # Initialize camera with context manager
        with RealSenseReader(width=640, height=480, fps=30) as camera:
            print("Camera started successfully!")
            print("Waiting for frames...")
            
            # Get intrinsics
            color_intr, depth_intr = camera.get_intrinsics()
            print(f"\nColor intrinsics: fx={color_intr.fx:.2f}, fy={color_intr.fy:.2f}, "
                  f"ppx={color_intr.ppx:.2f}, ppy={color_intr.ppy:.2f}")
            
            print("\n=== CONTROLS ===")
            print("Press 'q' to quit")
            print("Press 's' to save images")
            print("Press 'c' to detect chessboard")
            print("Detecting hand wrist keypoints...\n")
            
            frame_count = 0
            chessboard_origin = None
            chessboard_detected = False
            
            # Create window
            cv2.namedWindow('RealSense - RGB + Hand Keypoints (Left) | Depth + Keypoints (Right)', cv2.WINDOW_NORMAL)
            
            # Initialize MediaPipe Hands
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                
                print("Starting main loop...")
                
                while True:
                    # Read images
                    rgb_image, depth_image, depth_image_mm = camera.read()
                    
                    if rgb_image is None:
                        print("Warning: No frame received")
                        continue
                    
                    if frame_count == 0:
                        print(f"First frame received! Shape: {rgb_image.shape}")
                    
                    # Create working copies
                    display_rgb = rgb_image.copy()
                    
                    # Convert BGR to RGB for MediaPipe
                    rgb_for_mp = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    
                    # Process hand detection
                    results = hands.process(rgb_for_mp)
                    
                    # Create depth colormap for visualization with better range
                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    depth_normalized = np.uint8(depth_normalized)
                    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    
                    # Add depth range information on the colormap
                    min_depth = np.min(depth_image_mm[depth_image_mm > 0]) if np.any(depth_image_mm > 0) else 0
                    max_depth = np.max(depth_image_mm)
                    cv2.putText(depth_colormap, f"Min: {min_depth:.0f}mm", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(depth_colormap, f"Max: {max_depth:.0f}mm", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw chessboard origin if detected and calibration is loaded
                    if chessboard_detected and calibration_loaded and chessboard_origin is not None:
                        origin_x, origin_y = chessboard_origin
                        
                        # Draw coordinate axes at origin on RGB image
                        axis_length = 50
                        cv2.drawFrameAxes(display_rgb, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
                        
                        # Draw origin point (large marker)
                        cv2.circle(display_rgb, (origin_x, origin_y), 10, (0, 0, 255), -1)
                        cv2.circle(display_rgb, (origin_x, origin_y), 15, (255, 255, 255), 2)
                        cv2.putText(display_rgb, "Origin (0,0,0)", (origin_x + 20, origin_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Draw on depth image too
                        cv2.circle(depth_colormap, (origin_x, origin_y), 10, (0, 0, 255), -1)
                        cv2.circle(depth_colormap, (origin_x, origin_y), 15, (255, 255, 255), 2)
                        cv2.putText(depth_colormap, "Origin", (origin_x + 20, origin_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Process hand landmarks if detected
                    if results.multi_hand_landmarks:
                        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            # Extract wrist landmark (index 0)
                            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                            h, w = rgb_image.shape[:2]
                            
                            # Convert normalized coordinates to pixel coordinates
                            wrist_x = int(wrist.x * w)
                            wrist_y = int(wrist.y * h)
                            
                            # Ensure coordinates are within bounds
                            wrist_x = max(0, min(wrist_x, w - 1))
                            wrist_y = max(0, min(wrist_y, h - 1))
                            
                            # Get depth at wrist location (in mm)
                            wrist_depth_mm = depth_image_mm[wrist_y, wrist_x]
                            
                            # Draw wrist keypoint on RGB image with larger circle
                            cv2.circle(display_rgb, (wrist_x, wrist_y), 8, (0, 255, 255), -1)
                            cv2.circle(display_rgb, (wrist_x, wrist_y), 12, (255, 255, 0), 2)
                            
                            # Draw wrist keypoint on depth image
                            cv2.circle(depth_colormap, (wrist_x, wrist_y), 8, (255, 255, 255), -1)
                            cv2.circle(depth_colormap, (wrist_x, wrist_y), 12, (0, 255, 255), 2)
                            
                            # Calculate wrist position in chessboard coordinates
                            if calibration_loaded and transformer and wrist_depth_mm > 0:
                                try:
                                    # Transform to chessboard coordinates
                                    chess_coords = transformer.pixel_to_chessboard_coords(
                                        wrist_x, wrist_y, wrist_depth_mm)
                                    
                                    # Calculate distance from origin
                                    dist_from_origin = transformer.distance_from_origin(chess_coords)
                                    
                                    # Display coordinates on RGB image
                                    text_y_offset = -20 - (hand_idx * 140)
                                    cv2.putText(display_rgb, f"=== Hand {hand_idx + 1} ===", 
                                               (wrist_x - 60, wrist_y + text_y_offset),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    
                                    # Camera frame
                                    cv2.putText(display_rgb, f"Depth: {wrist_depth_mm:.1f}mm", 
                                               (wrist_x - 60, wrist_y + text_y_offset + 25),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                                    
                                    # Chessboard frame
                                    cv2.putText(display_rgb, f"Chess X: {chess_coords[0]:.1f}mm", 
                                               (wrist_x - 60, wrist_y + text_y_offset + 50),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                                    cv2.putText(display_rgb, f"Chess Y: {chess_coords[1]:.1f}mm", 
                                               (wrist_x - 60, wrist_y + text_y_offset + 70),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                                    cv2.putText(display_rgb, f"Chess Z: {chess_coords[2]:.1f}mm", 
                                               (wrist_x - 60, wrist_y + text_y_offset + 90),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                                    cv2.putText(display_rgb, f"Dist: {dist_from_origin:.1f}mm", 
                                               (wrist_x - 60, wrist_y + text_y_offset + 115),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
                                    
                                    # Display on depth image
                                    cv2.putText(depth_colormap, f"Wrist {hand_idx + 1}: {wrist_depth_mm:.0f}mm", 
                                               (wrist_x - 70, wrist_y - 40),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                                    cv2.putText(depth_colormap, f"Chess: ({chess_coords[0]:.0f}, {chess_coords[1]:.0f}, {chess_coords[2]:.0f})", 
                                               (wrist_x - 70, wrist_y - 20),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                                    
                                except Exception as e:
                                    # Fallback to simple display
                                    text_y_offset = -20 - (hand_idx * 60)
                                    cv2.putText(display_rgb, f"Wrist {hand_idx + 1}", 
                                               (wrist_x - 50, wrist_y + text_y_offset),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    cv2.putText(display_rgb, f"Depth: {wrist_depth_mm:.1f}mm", 
                                               (wrist_x - 50, wrist_y + text_y_offset + 25),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(depth_colormap, f"Wrist: {wrist_depth_mm:.1f}mm", 
                                               (wrist_x - 50, wrist_y - 30),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            else:
                                # Simple display without coordinate transform
                                text_y_offset = -20 - (hand_idx * 60)
                                cv2.putText(display_rgb, f"Wrist {hand_idx + 1}", 
                                           (wrist_x - 50, wrist_y + text_y_offset),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                cv2.putText(display_rgb, f"Depth: {wrist_depth_mm:.1f}mm", 
                                           (wrist_x - 50, wrist_y + text_y_offset + 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(depth_colormap, f"Wrist: {wrist_depth_mm:.1f}mm", 
                                           (wrist_x - 50, wrist_y - 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Stack images horizontally for display
                    images = np.hstack((display_rgb, depth_colormap))
                    
                    # Add frame counter
                    cv2.putText(images, f"Frame: {frame_count}", (10, images.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display
                    cv2.imshow('RealSense - RGB + Hand Keypoints (Left) | Depth + Keypoints (Right)', images)
                    
                    frame_count += 1
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('s'):
                        # Save images
                        save_count = len([f for f in os.listdir('capture') if f.startswith('rgb_')])
                        cv2.imwrite(f'capture/rgb_{save_count:04d}.png', display_rgb)
                        cv2.imwrite(f'capture/depth_{save_count:04d}.png', depth_image)
                        cv2.imwrite(f'capture/depth_colormap_{save_count:04d}.png', depth_colormap)
                        np.save(f'capture/depth_{save_count:04d}.npy', depth_image_mm)
                        print(f"✓ Saved frame {save_count} to capture/")
                    elif key == ord('c'):
                        # Detect chessboard
                        if calibration_loaded:
                            print("Detecting chessboard...")
                            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
                            
                            if ret:
                                # Refine corner positions
                                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                                
                                # Prepare object points
                                objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
                                objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
                                
                                # Get pose
                                ret, rvec_new, tvec_new = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)
                                
                                if ret:
                                    # Update rvec and tvec
                                    rvec = rvec_new
                                    tvec = tvec_new
                                    
                                    # Get origin point (bottom-left corner)
                                    chessboard_origin = get_chessboard_origin_2D(rvec, tvec, camera_matrix, dist_coeffs, CHESSBOARD_SIZE, SQUARE_SIZE)
                                    chessboard_detected = True
                                    print(f"✓ Chessboard detected! Origin at: {chessboard_origin}")
                                    print(f"  Translation: {tvec.flatten()}")
                                else:
                                    print("✗ Could not solve PnP")
                            else:
                                print("✗ Chessboard not found in frame")
                        else:
                            print("✗ Calibration data not loaded")
            
            cv2.destroyAllWindows()
            print("Program finished successfully!")
            
    except Exception as e:
        print(f"\n!!! ERROR !!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
