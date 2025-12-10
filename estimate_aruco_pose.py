"""
ArUco Pose Estimation Tool

Detect ArUco markers and estimate their pose (position and orientation) in 3D space.
Useful for testing ArUco detection and setting up coordinate frames.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from marker_config import *


class ArucoPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        """
        Initialize ArUco pose estimator
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Initialize ArUco detector
        self.aruco_dict = get_aruco_dictionary()
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        print(f"✓ ArUco Pose Estimator initialized")
        print(f"  Dictionary: {ARUCO_DICT}")
        print(f"  Marker size: {ARUCO_MARKER_SIZE} mm")
    
    def detect_and_estimate_pose(self, image):
        """
        Detect ArUco markers and estimate their poses
        
        Args:
            image: Input BGR image
        
        Returns:
            results: List of dictionaries containing marker data
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return []
        
        results = []
        
        # Estimate pose for each detected marker
        for i, marker_id in enumerate(ids):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], 
                ARUCO_MARKER_SIZE, 
                self.camera_matrix, 
                self.dist_coeffs
            )
            
            rvec = rvecs[0]
            tvec = tvecs[0]
            distance = np.linalg.norm(tvec)
            R, _ = cv2.Rodrigues(rvec)
            
            # Calculate Euler angles
            sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
            singular = sy < 1e-6
            
            if not singular:
                roll = np.arctan2(R[2,1], R[2,2])
                pitch = np.arctan2(-R[2,0], sy)
                yaw = np.arctan2(R[1,0], R[0,0])
            else:
                roll = np.arctan2(-R[1,2], R[1,1])
                pitch = np.arctan2(-R[2,0], sy)
                yaw = 0
            
            results.append({
                'id': int(marker_id[0]),
                'corners': corners[i],
                'rvec': rvec,
                'tvec': tvec,
                'distance': distance,
                'rotation_matrix': R,
                'roll': np.degrees(roll),
                'pitch': np.degrees(pitch),
                'yaw': np.degrees(yaw)
            })
        
        return results
    
    def draw_markers_and_axes(self, image, results):
        """Draw detected markers and coordinate axes"""
        output = image.copy()
        
        for result in results:
            marker_id = result['id']
            corners = result['corners']
            rvec = result['rvec']
            tvec = result['tvec']
            distance = result['distance']
            
            # Draw marker
            cv2.aruco.drawDetectedMarkers(output, [corners], np.array([[marker_id]]))
            
            # Draw axes
            axis_length = ARUCO_MARKER_SIZE * 0.5
            cv2.drawFrameAxes(output, self.camera_matrix, self.dist_coeffs, 
                            rvec, tvec, axis_length)
            
            # Draw axis labels
            label_offset = 1.3
            label_points = np.float32([[0, 0, 0],
                                      [axis_length * label_offset, 0, 0],
                                      [0, axis_length * label_offset, 0],
                                      [0, 0, axis_length * label_offset]])
            
            label_img_points, _ = cv2.projectPoints(label_points, rvec, tvec,
                                                   self.camera_matrix, self.dist_coeffs)
            label_img_points = label_img_points.reshape(-1, 2).astype(int)
            
            cv2.putText(output, 'X', tuple(label_img_points[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(output, 'Y', tuple(label_img_points[2]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output, 'Z', tuple(label_img_points[3]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw info box
            center = corners[0].mean(axis=0).astype(int)
            info_lines = [
                f"ID: {marker_id}",
                f"Dist: {distance:.1f}mm",
                f"X: {tvec[0][0]:.1f}mm",
                f"Y: {tvec[0][1]:.1f}mm",
                f"Z: {tvec[0][2]:.1f}mm",
            ]
            
            box_y = center[1] - 20
            for i, line in enumerate(info_lines):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(output, 
                            (center[0] + 20, box_y + i*20 - 15),
                            (center[0] + 25 + text_size[0], box_y + i*20 + 5),
                            (0, 0, 0), -1)
                cv2.putText(output, line, 
                           (center[0] + 22, box_y + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return output
    
    def print_pose_info(self, results):
        """Print detailed pose information"""
        if not results:
            print("No markers detected")
            return
        
        print("\n" + "="*70)
        print(f"Detected {len(results)} marker(s)")
        print("="*70)
        
        for i, result in enumerate(results):
            print(f"\nMarker #{i+1}")
            print(f"  ID: {result['id']}")
            print(f"  Position (mm):")
            print(f"    X: {result['tvec'][0][0]:8.2f}")
            print(f"    Y: {result['tvec'][0][1]:8.2f}")
            print(f"    Z: {result['tvec'][0][2]:8.2f}")
            print(f"  Distance: {result['distance']:.2f} mm")
            print(f"  Orientation (degrees):")
            print(f"    Roll:  {result['roll']:7.2f}°")
            print(f"    Pitch: {result['pitch']:7.2f}°")
            print(f"    Yaw:   {result['yaw']:7.2f}°")
        
        print("="*70 + "\n")


def estimate_from_realsense():
    """Estimate ArUco pose using RealSense camera"""
    
    print("="*70)
    print("ArUco Pose Estimation with RealSense")
    print("="*70)
    print_config()
    
    # Load camera calibration
    try:
        camera_matrix = np.load(f"{MATRIX_DIR}camera_matrix.npy")
        dist_coeffs = np.load(f"{MATRIX_DIR}dist_coeffs.npy")
        print("\n✓ Loaded camera calibration from matrix/")
    except Exception as e:
        print(f"\n✗ Error loading calibration: {e}")
        print("Run realsense_calibrate.py first!")
        return
    
    # Initialize pose estimator
    estimator = ArucoPoseEstimator(camera_matrix, dist_coeffs)
    
    # Initialize RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("\n✓ Starting RealSense camera...")
    profile = pipeline.start(config)
    
    # Get intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    print(f"✓ Camera intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, "
          f"ppx={intr.ppx:.2f}, ppy={intr.ppy:.2f}")
    
    print("\n=== CONTROLS ===")
    print("  'Space' - Print detailed pose info")
    print("  's' - Save current frame")
    print("  'c' - Save camera parameters")
    print("  'q' - Quit")
    print()
    
    # Create folders
    calib_dir = 'calibration/'
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
    if not os.path.exists(CAPTURE_DIR):
        os.makedirs(CAPTURE_DIR)
    
    cv2.namedWindow('ArUco Pose Estimation', cv2.WINDOW_NORMAL)
    
    saved_count = 0
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            image = np.asanyarray(color_frame.get_data())
            results = estimator.detect_and_estimate_pose(image)
            output = estimator.draw_markers_and_axes(image, results)
            
            # Add status
            status_text = f"Detected: {len(results)} marker(s)"
            cv2.putText(output, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if ARUCO_SINGLE_MARKER:
                target_text = f"Target ID: {ARUCO_SINGLE_ID}"
                cv2.putText(output, target_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('ArUco Pose Estimation', output)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                estimator.print_pose_info(results)
            elif key == ord('s'):
                if results:
                    filename = f'{CAPTURE_DIR}aruco_pose_{saved_count:04d}.png'
                    cv2.imwrite(filename, output)
                    print(f"✓ Saved: {filename}")
                    saved_count += 1
            elif key == ord('c'):
                if results:
                    # Save calibration parameters
                    np.save(f'{calib_dir}camera_matrix.npy', camera_matrix)
                    np.save(f'{calib_dir}dist_coeffs.npy', dist_coeffs)
                    
                    result = results[0]
                    np.save(f'{calib_dir}rvec.npy', result['rvec'])
                    np.save(f'{calib_dir}tvec.npy', result['tvec'])
                    np.save(f'{calib_dir}rotation_matrix.npy', result['rotation_matrix'])
                    
                    T = np.eye(4)
                    T[:3, :3] = result['rotation_matrix']
                    T[:3, 3] = result['tvec'].flatten()
                    np.save(f'{calib_dir}T_camera_to_marker.npy', T)
                    
                    with open(f'{calib_dir}camera_parameters.txt', 'w') as f:
                        f.write("="*60 + "\n")
                        f.write("CAMERA CALIBRATION PARAMETERS\n")
                        f.write("="*60 + "\n\n")
                        f.write("INTRINSIC PARAMETERS\n")
                        f.write("-"*60 + "\n")
                        f.write("Camera Matrix:\n")
                        f.write(f"{camera_matrix}\n\n")
                        f.write("Distortion Coefficients:\n")
                        f.write(f"{dist_coeffs}\n\n")
                        f.write("EXTRINSIC PARAMETERS\n")
                        f.write("-"*60 + "\n")
                        f.write(f"Marker ID: {result['id']}\n")
                        f.write(f"Position: {result['tvec'].flatten()}\n")
                        f.write(f"Rotation: {result['rvec'].flatten()}\n")
                        f.write(f"Distance: {result['distance']:.2f} mm\n")
                    
                    print(f"\n✓ Saved camera parameters to {calib_dir}")
                else:
                    print("✗ No markers detected")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n✓ Program finished!")


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Processing image: {image_path}")
        # Add image processing here if needed
    else:
        estimate_from_realsense()


if __name__ == "__main__":
    main()
