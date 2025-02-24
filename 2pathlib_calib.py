###########                                     sensor fusion                         #############################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from struct import unpack

# Load calibration parameters from a KITTI calibration file
def load_calibration(calib_file):
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file {calib_file} not found.")
        return None, None, None
    
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    def parse_matrix(line):
        return np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
    
    P2 = parse_matrix(lines[2])  # Camera projection matrix
    
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
    
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)
    
    return P2, R0_rect, Tr_velo_to_cam

# Load LiDAR point cloud from a binary file (.bin)
def load_lidar_data(bin_file):
    if not os.path.exists(bin_file):
        print(f"Error: LiDAR file {bin_file} not found.")
        return np.array([])
    
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

# Project LiDAR points onto the image plane
def project_lidar_to_image(lidar_points, P2, R0_rect, Tr_velo_to_cam):
    if lidar_points.size == 0:
        print("Error: No LiDAR points loaded.")
        return np.array([]), np.array([])
    
    lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    cam_coords = (R0_rect @ Tr_velo_to_cam @ lidar_hom.T).T
    cam_coords = cam_coords[cam_coords[:, 2] > 0]
    img_coords = (P2 @ cam_coords.T).T
    img_coords[:, 0] /= img_coords[:, 2]
    img_coords[:, 1] /= img_coords[:, 2]
    return img_coords[:, :2], cam_coords[:, 2]

# Overlay projected LiDAR points on the image
def overlay_lidar_on_image(image_path, img_coords, depths):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to open image file.")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for (x, y), depth in zip(img_coords, depths):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            color = (0, int(255 * (1 - depth / max(depths))), int(255 * (depth / max(depths))))
            cv2.circle(img, (int(x), int(y)), 2, color, -1)
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# File paths for input data
calib_file = "data/calib/000031.txt"
bin_file = "data/velodyne/000031.bin"
image_file = "data/img/000031.png"

# Debugging: Check if files exist
print("Checking file paths...")
print(f"Calibration file exists: {os.path.exists(calib_file)}")
print(f"LiDAR file exists: {os.path.exists(bin_file)}")
print(f"Image file exists: {os.path.exists(image_file)}")

# Load calibration parameters, LiDAR points, and project them onto the image
P2, R0_rect, Tr_velo_to_cam = load_calibration(calib_file)
if P2 is not None:
    lidar_points = load_lidar_data(bin_file)
    img_coords, depths = project_lidar_to_image(lidar_points, P2, R0_rect, Tr_velo_to_cam)
    
    if img_coords.size > 0:
        overlay_lidar_on_image(image_file, img_coords, depths)
