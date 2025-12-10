"""
RealSense Hand Tracking with ArUco Marker Origin

Real-time hand tracking using RealSense camera with ArUco marker as coordinate origin.
Similar to realsense_reader.py but uses ArUco marker instead of chessboard.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import os
from marker_config import *


class RealSenseReaderAruco:
    def __init__(self):
        """Initialize RealSense camera and ArUco detector"""
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        print("Starting RealSense camera...")
        self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        
        # Create camera matrix
        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(self.intrinsics.coeffs)
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
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
        
        print("✓ Camera initialized")
        print(f"✓ Intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
              f"ppx={self.intrinsics.ppx:.2f}, ppy={self.intrinsics.ppy:.2f}")
        print(f"✓ ArUco detector ready ({ARUCO_DICT})")
        print(f"✓ MediaPipe Hands ready")
    
    def detect_aruco(self, image):
        """Detect ArUco marker and estimate pose"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return None
        
        # Filter for target marker if single marker mode
        if ARUCO_SINGLE_MARKER:
            target_idx = None
            for i, marker_id in enumerate(ids):
                if marker_id[0] == ARUCO_SINGLE_ID:
                    target_idx = i
                    break
            
            if target_idx is None:
                return None
            
            corners = [corners[target_idx]]
            ids = [ids[target_idx]]
        
        # Estimate pose for first (or target) marker
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
    
    def smooth_depth(self, depth_image):
        """Apply Gaussian smoothing to depth image"""
        return cv2.GaussianBlur(depth_image, (5, 5), 0)
    
    def fill_zero_depth(self, depth_image, iterations=3):
        """Fill zero depth values using neighbor averaging"""
        result = depth_image.copy().astype(np.float32)
        
        for _ in range(iterations):
            mask = (result == 0)
            if not mask.any():
                break
            
            kernel = np.ones((3, 3), np.float32) / 9
            filled = cv2.filter2D(result, -1, kernel)
            result[mask] = filled[mask]
        
        return result.astype(np.uint16)
    
    def get_frames(self):
        """Get aligned color and depth frames"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Process depth
        depth_image = self.fill_zero_depth(depth_image)
        depth_image = self.smooth_depth(depth_image)
        
        return color_image, depth_image, depth_frame
    
    def detect_hands(self, color_image):
        """Detect hands using MediaPipe"""
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_data = []
        if results.multi_hand_landmarks:
            h, w = color_image.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                hand_data.append({
                    'wrist_px': wrist_px,
                    'landmarks': hand_landmarks
                })
        
        return hand_data
    
    def pixel_to_camera_coords(self, pixel, depth):
        """Convert pixel + depth to camera 3D coordinates"""
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
    
    def visualize(self, color_image, depth_image, depth_frame):
        """Create visualization with marker and hand tracking"""
        output_color = color_image.copy()
        output_depth = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        # Detect ArUco marker
        marker_data = self.detect_aruco(color_image)
        
        # Detect hands
        hand_data = self.detect_hands(color_image)
        
        # Draw marker if detected
        if marker_data:
            # Draw marker boundaries
            cv2.aruco.drawDetectedMarkers(output_color, [marker_data['corners']], 
                                         np.array([[marker_data['id']]]))
            cv2.aruco.drawDetectedMarkers(output_depth, [marker_data['corners']], 
                                         np.array([[marker_data['id']]]))
            
            # Draw coordinate axes
            axis_length = ARUCO_MARKER_SIZE * 0.5
            cv2.drawFrameAxes(output_color, self.camera_matrix, self.dist_coeffs,
                            marker_data['rvec'], marker_data['tvec'], axis_length)
            cv2.drawFrameAxes(output_depth, self.camera_matrix, self.dist_coeffs,
                            marker_data['rvec'], marker_data['tvec'], axis_length)
            
            # Get marker origin in image
            origin_3d = np.array([[0, 0, 0]], dtype=np.float32)
            origin_2d, _ = cv2.projectPoints(origin_3d, marker_data['rvec'], 
                                            marker_data['tvec'],
                                            self.camera_matrix, self.dist_coeffs)
            origin_px = tuple(origin_2d[0][0].astype(int))
            
            cv2.circle(output_color, origin_px, 8, (0, 255, 0), -1)
            cv2.circle(output_depth, origin_px, 8, (0, 255, 0), -1)
        
        # Draw hands
        for hand in hand_data:
            wrist_px = hand['wrist_px']
            
            # Draw wrist on both images
            cv2.circle(output_color, wrist_px, 8, (0, 255, 255), -1)
            cv2.circle(output_depth, wrist_px, 8, (0, 255, 255), -1)
            
            # Get depth and calculate coordinates if marker detected
            if marker_data and depth_frame:
                depth = depth_frame.get_distance(wrist_px[0], wrist_px[1]) * 1000  # mm
                
                if depth > 0:
                    wrist_camera = self.pixel_to_camera_coords(wrist_px, depth)
                    wrist_marker = self.camera_to_marker_coords(wrist_camera, marker_data)
                    
                    # Draw info
                    info = f"({wrist_marker[0]:.1f}, {wrist_marker[1]:.1f}, {wrist_marker[2]:.1f})"
                    cv2.putText(output_color, info, (wrist_px[0] + 10, wrist_px[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add status
        status = f"Marker: {'OK' if marker_data else 'None'} | Hands: {len(hand_data)}"
        cv2.putText(output_color, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_depth, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output_color, output_depth
    
    def run(self):
        """Main loop"""
        print("\n=== CONTROLS ===")
        print("  'q' - Quit")
        print()
        
        cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                color_image, depth_image, depth_frame = self.get_frames()
                
                if color_image is None:
                    continue
                
                output_color, output_depth = self.visualize(color_image, depth_image, depth_frame)
                
                cv2.imshow('RGB', output_color)
                cv2.imshow('Depth', output_depth)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("\n✓ Program finished!")


def main():
    print("="*70)
    print("RealSense Hand Tracking with ArUco Marker")
    print("="*70)
    print_config()
    
    reader = RealSenseReaderAruco()
    reader.run()


if __name__ == "__main__":
    main()
