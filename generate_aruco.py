"""
Generate ArUco Markers for Printing

Generate ArUco markers from DICT_4X4_50 dictionary for calibration and tracking.
"""

import cv2
import numpy as np
from marker_config import *


def generate_single_marker(marker_id, size_pixels=400, border_bits=1, save_path=None):
    """
    Generate a single ArUco marker
    
    Args:
        marker_id: Marker ID to generate
        size_pixels: Size of marker in pixels (default 400)
        border_bits: Border size in bits (default 1)
        save_path: Path to save marker image (optional)
    
    Returns:
        Marker image as numpy array
    """
    aruco_dict = get_aruco_dictionary()
    
    # Generate marker
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels, borderBits=border_bits)
    
    if save_path:
        cv2.imwrite(save_path, marker_img)
        print(f"✓ Saved marker ID {marker_id} to {save_path}")
    
    return marker_img


def generate_marker_board(marker_ids, cols, rows, marker_size_pixels=200, 
                          spacing_pixels=50, border_bits=1, save_path=None):
    """
    Generate a board with multiple ArUco markers
    
    Args:
        marker_ids: List of marker IDs to generate
        cols: Number of columns
        rows: Number of rows
        marker_size_pixels: Size of each marker in pixels
        spacing_pixels: Spacing between markers in pixels
        border_bits: Border size in bits
        save_path: Path to save board image (optional)
    
    Returns:
        Board image as numpy array
    """
    aruco_dict = get_aruco_dictionary()
    
    if len(marker_ids) != cols * rows:
        raise ValueError(f"Number of markers ({len(marker_ids)}) must equal cols * rows ({cols * rows})")
    
    # Calculate board dimensions
    board_width = cols * marker_size_pixels + (cols + 1) * spacing_pixels
    board_height = rows * marker_size_pixels + (rows + 1) * spacing_pixels
    
    # Create white board
    board_img = np.ones((board_height, board_width), dtype=np.uint8) * 255
    
    # Place markers on board
    idx = 0
    for row in range(rows):
        for col in range(cols):
            marker_id = marker_ids[idx]
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 
                                                       marker_size_pixels, borderBits=border_bits)
            
            # Calculate position
            x = spacing_pixels + col * (marker_size_pixels + spacing_pixels)
            y = spacing_pixels + row * (marker_size_pixels + spacing_pixels)
            
            # Place marker
            board_img[y:y+marker_size_pixels, x:x+marker_size_pixels] = marker_img
            
            idx += 1
    
    if save_path:
        cv2.imwrite(save_path, board_img)
        print(f"✓ Saved marker board to {save_path}")
    
    return board_img


def main():
    """Generate ArUco markers based on configuration"""
    print("="*70)
    print("ArUco Marker Generator")
    print("="*70)
    print_config()
    
    if ARUCO_SINGLE_MARKER:
        # Generate single marker
        print(f"\nGenerating single marker ID {ARUCO_SINGLE_ID}...")
        marker_img = generate_single_marker(
            marker_id=ARUCO_SINGLE_ID,
            size_pixels=800,
            border_bits=1,
            save_path=f'aruco_marker_{ARUCO_SINGLE_ID}.png'
        )
        
        # Display
        cv2.imshow(f'ArUco Marker ID {ARUCO_SINGLE_ID}', marker_img)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"\n✓ Marker saved as: aruco_marker_{ARUCO_SINGLE_ID}.png")
        print(f"✓ Print this at {ARUCO_MARKER_SIZE}mm x {ARUCO_MARKER_SIZE}mm")
    else:
        # Generate marker board
        print("\nGenerating marker board...")
        marker_ids = [0, 1, 2, 3]
        board_img = generate_marker_board(
            marker_ids=marker_ids,
            cols=2,
            rows=2,
            marker_size_pixels=300,
            spacing_pixels=50,
            border_bits=1,
            save_path='aruco_board.png'
        )
        
        # Display
        cv2.imshow('ArUco Marker Board', board_img)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"\n✓ Board saved as: aruco_board.png")
        print(f"✓ Each marker should be {ARUCO_MARKER_SIZE}mm x {ARUCO_MARKER_SIZE}mm")


if __name__ == "__main__":
    main()
