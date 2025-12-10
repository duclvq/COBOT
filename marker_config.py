"""
Configuration for calibration markers (Chessboard or ArUco)
"""

# ============================================================================
# MARKER TYPE
# ============================================================================
MARKER_TYPE = 'chessboard'  # 'chessboard' or 'aruco'

# ============================================================================
# CHESSBOARD CONFIGURATION
# ============================================================================
CHESSBOARD_SIZE = (9, 7)  # (columns, rows) of internal corners
SQUARE_SIZE = 20  # Size of each square in mm

# ============================================================================
# ARUCO CONFIGURATION
# ============================================================================
ARUCO_DICT = 'DICT_4X4_50'  # 4x4 dictionary
ARUCO_MARKER_SIZE = 95  # Physical size of printed marker in mm
ARUCO_SINGLE_MARKER = True  # Use single marker mode
ARUCO_SINGLE_ID = 0  # Marker ID 0

# ============================================================================
# COMMON SETTINGS
# ============================================================================
MATRIX_DIR = 'matrix/'
CAPTURE_DIR = 'capture/'

# ============================================================================
# FUNCTIONS
# ============================================================================

def print_config():
    """Print current configuration."""
    print("\n" + "="*60)
    print(f"Marker Type: {MARKER_TYPE.upper()}")
    print("="*60)
    
    if MARKER_TYPE == 'chessboard':
        print(f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} corners")
        print(f"Square Size: {SQUARE_SIZE} mm")
    elif MARKER_TYPE == 'aruco':
        print(f"ArUco Dictionary: {ARUCO_DICT}")
        print(f"Marker ID: {ARUCO_SINGLE_ID}")
        print(f"Marker Size: {ARUCO_MARKER_SIZE} mm")
    
    print("="*60 + "\n")


def get_aruco_dictionary():
    """Get ArUco dictionary object."""
    import cv2
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


if __name__ == "__main__":
    print_config()
