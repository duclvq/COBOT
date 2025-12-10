"""
Track Hand Position Relative to ArUco Marker

Track wrist position relative to ArUco marker coordinate system.
Display dashed line from marker origin to wrist position.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import os
from marker_config import *


class HandMarkerTracker:
    def __init__(self, camera_matrix, dist_coeffs):
        """Initialize hand-marker tracker"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Initialize ArUco detector
        self.aruco_dict = get_aruco_dictionary()
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"✓ Hand-Marker Tracker initialized")
        print(f"  ArUco: {ARUCO_DICT}, Marker size: {ARUCO_MARKER_SIZE} mm")
        print(f"  MediaPipe Hands: max 2 hands, confidence 0.5")
    
    def detect_aruco(self, image):
        """Detect ArUco marker and estimate pose"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return None
        
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[0]], 
            ARUCO_MARKER_SIZE, 
            self.camera_matrix, 
            self.dist_coeffs
        )
        
        rvec = rvecs[0]
        tvec = tvecs[0]
        R, _ = cv2.Rodrigues(rvec)
        
        return {
            'id': int(ids[0][0]),
            'corners': corners[0],
            'rvec': rvec,
            'tvec': tvec,
            'R': R
        }
    
    def detect_hands(self, image):
        """Detect hand wrist positions"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        wrists = []
        if results.multi_hand_landmarks:
            h, w = image.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                wrists.append(wrist_px)
        
        return wrists
    
    def pixel_to_camera_coords(self, pixel, depth):
        """Convert pixel coordinates to camera 3D coordinates"""
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        x = (pixel[0] - cx) * depth / fx
        y = (pixel[1] - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def camera_to_marker_coords(self, point_camera, marker_data):
        """Transform point from camera frame to marker frame"""
        R = marker_data['R']
        tvec = marker_data['tvec'].flatten()
        
        point_rel = point_camera - tvec
        point_marker = R.T @ point_rel
        
        return point_marker
    
    def draw_dashed_line(self, image, pt1, pt2, color, thickness=2, dash_length=10):
        """Draw a dashed line between two points"""
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        
        dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
        dashes = int(dist / dash_length)
        
        if dashes == 0:
            cv2.line(image, pt1, pt2, color, thickness)
            return
        
        for i in range(0, dashes, 2):
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)
            
            start_pt = (
                int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
            )
            end_pt = (
                int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
                int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
            )
            
            cv2.line(image, start_pt, end_pt, color, thickness)
    
    def visualize(self, color_image, depth_frame, marker_data, wrists):
        """Visualize marker, hands, and connections"""
        output = color_image.copy()
        
        if marker_data is None:
            cv2.putText(output, "No marker detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output, []
        
        # Draw marker
        cv2.aruco.drawDetectedMarkers(output, [marker_data['corners']], 
                                     np.array([[marker_data['id']]]))
        
        # Draw coordinate axes
        axis_length = ARUCO_MARKER_SIZE * 0.5
        cv2.drawFrameAxes(output, self.camera_matrix, self.dist_coeffs, 
                         marker_data['rvec'], marker_data['tvec'], axis_length)
        
        # Draw axis labels
        label_offset = 1.3
        label_points = np.float32([[0, 0, 0],
                                  [axis_length * label_offset, 0, 0],
                                  [0, axis_length * label_offset, 0],
                                  [0, 0, axis_length * label_offset]])
        
        label_img_points, _ = cv2.projectPoints(label_points, marker_data['rvec'], 
                                               marker_data['tvec'],
                                               self.camera_matrix, self.dist_coeffs)
        label_img_points = label_img_points.reshape(-1, 2).astype(int)
        
        cv2.putText(output, 'X', tuple(label_img_points[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output, 'Y', tuple(label_img_points[2]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, 'Z', tuple(label_img_points[3]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Get marker origin in image
        origin_3d = np.array([[0, 0, 0]], dtype=np.float32)
        origin_2d, _ = cv2.projectPoints(origin_3d, marker_data['rvec'], 
                                        marker_data['tvec'],
                                        self.camera_matrix, self.dist_coeffs)
        origin_px = tuple(origin_2d[0][0].astype(int))
        
        # Process each wrist
        hand_positions = []
        for idx, wrist_px in enumerate(wrists):
            # Check bounds
            h, w = color_image.shape[:2]
            if wrist_px[0] < 0 or wrist_px[0] >= w or wrist_px[1] < 0 or wrist_px[1] >= h:
                continue
            
            # Get depth
            depth = depth_frame.get_distance(wrist_px[0], wrist_px[1]) * 1000  # mm
            
            if depth > 0:
                # Convert to camera coordinates
                wrist_camera = self.pixel_to_camera_coords(wrist_px, depth)
                
                # Transform to marker coordinates
                wrist_marker = self.camera_to_marker_coords(wrist_camera, marker_data)
                
                # Draw wrist point
                cv2.circle(output, wrist_px, 8, (0, 255, 255), -1)
                cv2.circle(output, wrist_px, 10, (255, 255, 255), 2)
                
                # Draw dashed line
                self.draw_dashed_line(output, origin_px, wrist_px, 
                                    (255, 200, 0), thickness=2, dash_length=8)
                
                # Draw info
                info_lines = [
                    f"Hand {idx+1}",
                    f"X: {wrist_marker[0]:.1f}mm",
                    f"Y: {wrist_marker[1]:.1f}mm",
                    f"Z: {wrist_marker[2]:.1f}mm",
                    f"D: {np.linalg.norm(wrist_marker):.1f}mm"
                ]
                
                box_x = wrist_px[0] + 15
                box_y = wrist_px[1] - 30
                
                for i, line in enumerate(info_lines):
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(output,
                                (box_x, box_y + i*18 - 12),
                                (box_x + text_size[0] + 4, box_y + i*18 + 6),
                                (0, 0, 0), -1)
                    cv2.putText(output, line, (box_x + 2, box_y + i*18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                hand_positions.append({
                    'pixel': wrist_px,
                    'camera': wrist_camera,
                    'marker': wrist_marker,
                    'distance': np.linalg.norm(wrist_marker)
                })
        
        return output, hand_positions


def main():
    """Main tracking loop"""
    print("="*70)
    print("Hand Position Tracking Relative to ArUco Marker")
    print("="*70)
    print_config()
    
    # Load calibration
    try:
        camera_matrix = np.load(f"{MATRIX_DIR}camera_matrix.npy")
        dist_coeffs = np.load(f"{MATRIX_DIR}dist_coeffs.npy")
        print("\n✓ Loaded camera calibration from matrix/")
    except Exception as e:
        print(f"\n✗ Error loading calibration: {e}")
        print("Run realsense_calibrate.py first!")
        return
    
    # Initialize tracker
    tracker = HandMarkerTracker(camera_matrix, dist_coeffs)
    
    # Initialize RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    print("\n✓ Starting RealSense camera...")
    profile = pipeline.start(config)
    
    # Align depth to color
    align = rs.align(rs.stream.color)
    
    print("\n=== CONTROLS ===")
    print("  'Space' - Print hand positions")
    print("  's' - Save frame")
    print("  'q' - Quit")
    print()
    
    cv2.namedWindow('Hand-Marker Tracking', cv2.WINDOW_NORMAL)
    
    saved_count = 0
    last_marker_data = None  # Store last known marker position
    
    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            
            # Detect marker
            marker_data = tracker.detect_aruco(color_image)
            
            # Save or use last known position
            if marker_data is not None:
                last_marker_data = marker_data
            else:
                marker_data = last_marker_data
            
            # Detect hands
            wrists = tracker.detect_hands(color_image)
            
            # Visualize
            output, hand_positions = tracker.visualize(color_image, depth_frame, 
                                                      marker_data, wrists)
            
            # Add status
            marker_status = "OK" if tracker.detect_aruco(color_image) else "LOST (using last)"
            if marker_data is None:
                marker_status = "None"
            status = f"Marker: {marker_status} | Hands: {len(wrists)}"
            status_color = (0, 255, 0) if tracker.detect_aruco(color_image) else (0, 165, 255)
            cv2.putText(output, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Display
            cv2.imshow('Hand-Marker Tracking', output)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                if hand_positions:
                    print("\n" + "="*60)
                    print(f"Hand Positions (relative to marker ID {marker_data['id']})")
                    print("="*60)
                    for i, pos in enumerate(hand_positions):
                        print(f"\nHand {i+1}:")
                        print(f"  Marker coords: X={pos['marker'][0]:7.2f}, "
                              f"Y={pos['marker'][1]:7.2f}, Z={pos['marker'][2]:7.2f} mm")
                        print(f"  Distance: {pos['distance']:.2f} mm")
                    print("="*60 + "\n")
                else:
                    print("No hands detected")
            elif key == ord('s'):
                if not os.path.exists(CAPTURE_DIR):
                    os.makedirs(CAPTURE_DIR)
                
                filename = f'{CAPTURE_DIR}hand_track_{saved_count:04d}.png'
                cv2.imwrite(filename, output)
                print(f"✓ Saved: {filename}")
                
                if hand_positions:
                    txt_filename = f'{CAPTURE_DIR}hand_track_{saved_count:04d}.txt'
                    with open(txt_filename, 'w') as f:
                        f.write("Hand Tracking Data\n")
                        f.write(f"Marker ID: {marker_data['id']}\n")
                        f.write(f"Marker size: {ARUCO_MARKER_SIZE} mm\n\n")
                        
                        for i, pos in enumerate(hand_positions):
                            f.write(f"Hand {i+1}:\n")
                            f.write(f"  Pixel: {pos['pixel']}\n")
                            f.write(f"  Camera coords: {pos['camera']}\n")
                            f.write(f"  Marker coords: {pos['marker']}\n")
                            f.write(f"  Distance: {pos['distance']:.2f} mm\n\n")
                    
                    print(f"✓ Saved data: {txt_filename}")
                
                saved_count += 1
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n✓ Program finished!")


if __name__ == "__main__":
    main()
