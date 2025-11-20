import numpy as np
import cv2


class HandCoordinateTransform:
    """
    Transform hand coordinates from camera frame to chessboard coordinate system.
    """
    
    def __init__(self, matrix_dir='matrix/'):
        """
        Initialize coordinate transformer with calibration data.
        
        Args:
            matrix_dir: Directory containing calibration matrices
        """
        self.matrix_dir = matrix_dir
        self.load_calibration()
    
    def load_calibration(self):
        """Load camera calibration and extrinsic parameters."""
        try:
            self.camera_matrix = np.load(f'{self.matrix_dir}camera_matrix.npy')
            self.dist_coeffs = np.load(f'{self.matrix_dir}dist_coeffs.npy')
            self.rvec = np.load(f'{self.matrix_dir}rvec.npy')
            self.tvec = np.load(f'{self.matrix_dir}tvec.npy')
            self.R = np.load(f'{self.matrix_dir}R.npy')
            
            # Try to load transformation matrix
            try:
                self.T_cam_to_chess = np.load(f'{self.matrix_dir}T_cam_to_chess.npy')
            except:
                # Create it from R and tvec
                self.T_cam_to_chess = self._create_transform_matrix()
            
            print("✓ Calibration data loaded successfully")
            self._print_calibration_info()
            
        except Exception as e:
            print(f"✗ Error loading calibration: {e}")
            raise
    
    def _create_transform_matrix(self):
        """Create transformation matrix from rotation and translation."""
        # Camera to chessboard transformation
        T_chess_to_cam = np.eye(4)
        T_chess_to_cam[:3, :3] = self.R
        T_chess_to_cam[:3, 3] = self.tvec.flatten()
        
        # Invert to get camera to chessboard
        T_cam_to_chess = np.linalg.inv(T_chess_to_cam)
        
        return T_cam_to_chess
    
    def _print_calibration_info(self):
        """Print calibration information."""
        print("\nCamera Intrinsics:")
        print(f"  fx: {self.camera_matrix[0, 0]:.2f}")
        print(f"  fy: {self.camera_matrix[1, 1]:.2f}")
        print(f"  cx: {self.camera_matrix[0, 2]:.2f}")
        print(f"  cy: {self.camera_matrix[1, 2]:.2f}")
        print(f"\nChessboard origin (camera frame): {self.tvec.flatten()} mm")
    
    def pixel_to_camera_coords(self, u, v, depth_mm):
        """
        Convert pixel coordinates with depth to 3D camera coordinates.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            depth_mm: Depth in millimeters
        
        Returns:
            numpy array [X, Y, Z] in camera frame (mm)
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Convert to camera coordinates
        X_cam = (u - cx) * depth_mm / fx
        Y_cam = (v - cy) * depth_mm / fy
        Z_cam = depth_mm
        
        return np.array([X_cam, Y_cam, Z_cam])
    
    def camera_to_chessboard_coords(self, camera_point):
        """
        Transform point from camera frame to chessboard frame.
        
        Args:
            camera_point: [X, Y, Z] in camera frame (mm)
        
        Returns:
            numpy array [X, Y, Z] in chessboard frame (mm)
        """
        # Create homogeneous coordinates
        point_homo = np.append(camera_point, 1)
        
        # Transform to chessboard frame
        chess_point_homo = self.T_cam_to_chess @ point_homo
        
        return chess_point_homo[:3]
    
    def pixel_to_chessboard_coords(self, u, v, depth_mm):
        """
        Convert pixel coordinates with depth directly to chessboard coordinates.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            depth_mm: Depth in millimeters
        
        Returns:
            numpy array [X, Y, Z] in chessboard frame (mm)
        """
        # First convert to camera coordinates
        camera_point = self.pixel_to_camera_coords(u, v, depth_mm)
        
        # Then transform to chessboard coordinates
        chess_point = self.camera_to_chessboard_coords(camera_point)
        
        return chess_point
    
    def undistort_point(self, u, v):
        """
        Undistort a pixel point.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
        
        Returns:
            Tuple (u_undistorted, v_undistorted)
        """
        points = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs, 
                                         P=self.camera_matrix)
        return undistorted[0][0][0], undistorted[0][0][1]
    
    def get_chessboard_origin_2D(self, chessboard_size, square_size):
        """
        Get the 2D pixel coordinates of the chessboard origin (bottom-left corner).
        
        Args:
            chessboard_size: (columns, rows) tuple
            square_size: Size of each square in mm
        
        Returns:
            Tuple (u, v) of pixel coordinates
        """
        # Origin is at (0, rows*square_size, 0) in chessboard frame
        origin_3d = np.array([[0, chessboard_size[1] * square_size, 0]], dtype=np.float32)
        projected, _ = cv2.projectPoints(origin_3d, self.rvec, self.tvec, 
                                         self.camera_matrix, self.dist_coeffs)
        u = int(projected[0][0][0])
        v = int(projected[0][0][1])
        return (u, v)
    
    def distance_from_origin(self, chess_point):
        """
        Calculate Euclidean distance from chessboard origin.
        
        Args:
            chess_point: [X, Y, Z] in chessboard frame
        
        Returns:
            Distance in mm
        """
        return np.linalg.norm(chess_point)


def test_transform():
    """Test the coordinate transformation."""
    print("="*60)
    print("Testing Hand Coordinate Transform")
    print("="*60)
    
    try:
        transformer = HandCoordinateTransform()
        
        # Test with a sample point
        print("\n--- Test Transformation ---")
        u, v = 320, 240  # Center of 640x480 image
        depth = 500  # 500mm depth
        
        print(f"\nInput:")
        print(f"  Pixel: ({u}, {v})")
        print(f"  Depth: {depth} mm")
        
        # Camera coordinates
        cam_point = transformer.pixel_to_camera_coords(u, v, depth)
        print(f"\nCamera frame coordinates:")
        print(f"  X: {cam_point[0]:.2f} mm")
        print(f"  Y: {cam_point[1]:.2f} mm")
        print(f"  Z: {cam_point[2]:.2f} mm")
        
        # Chessboard coordinates
        chess_point = transformer.camera_to_chessboard_coords(cam_point)
        print(f"\nChessboard frame coordinates:")
        print(f"  X: {chess_point[0]:.2f} mm")
        print(f"  Y: {chess_point[1]:.2f} mm")
        print(f"  Z: {chess_point[2]:.2f} mm")
        
        # Distance from origin
        dist = transformer.distance_from_origin(chess_point)
        print(f"\nDistance from origin: {dist:.2f} mm")
        
        # Direct transformation
        chess_point_direct = transformer.pixel_to_chessboard_coords(u, v, depth)
        print(f"\nDirect transformation (should be same):")
        print(f"  X: {chess_point_direct[0]:.2f} mm")
        print(f"  Y: {chess_point_direct[1]:.2f} mm")
        print(f"  Z: {chess_point_direct[2]:.2f} mm")
        
        print("\n✓ Transform test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_transform()
