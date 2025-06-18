import cv2
import numpy as np


def order_points(pts):
    # Sort points by x-coordinate
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    
    # Grab left-most and right-most points
    left = x_sorted[:2, :]
    right = x_sorted[2:, :]
    
    # Sort left points by y-coordinate
    left = left[np.argsort(left[:, 1]), :]
    # Sort right points by y-coordinate
    right = right[np.argsort(right[:, 1]), :]
    
    # Return ordered points: tl, tr, br, bl
    return np.vstack([left[0], right[0], right[1], left[1]])
    

def find_rectangle_corners(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Find minimum area rectangle
    rect = cv2.minAreaRect(contour)
    
    # Get the 4 corners
    box = cv2.boxPoints(rect)
    box = np.int8(box)
    # Order the corners (optional, can be useful)
    # Typically: top-left, top-right, bottom-right, bottom-left
    box = order_points(box)
    # Get the center and angle
    center = rect[0]
    w, h = rect[1]
    if w < h:
        angle = rect[2] + 90  # OpenCV gives angle in (0, 90] range
    else:
        angle = rect[2]
    
    return rect, box, center, angle