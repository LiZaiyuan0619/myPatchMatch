# # R = np.eye(3)
# # T = np.zeros(3)
# R = [[0.99590467, -0.01363085, 0.08937612]
#      [0.01685576, 0.99923009, -0.03542757]
#      [-0.0888244, 0.03678899, 0.99536767]]
# T = [[-0.99900655][-0.01930007][-0.0401673]]
# print(R)
# print(T)

import numpy as np
import cv2
import open3d as o3d

def depth_to_pointcloud(depth_image, camera_intrinsics):
    """
    Convert a depth image to a 3D point cloud.

    Parameters:
    - depth_image: The depth image.
    - camera_intrinsics: Camera intrinsic parameters (fx, fy, cx, cy).

    Returns:
    - point_cloud: The resulting point cloud.
    """
    points = []
    h, w = depth_image.shape

    fx, fy, cx, cy = camera_intrinsics

    for v in range(h):
        for u in range(w):
            depth = depth_image[v, u]
            if depth > 0:  # Check if depth is valid
                z = depth
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])

    point_cloud = np.array(points)
    return point_cloud

# Load depth image
depth_image = cv2.imread('depth_map125.png', cv2.IMREAD_UNCHANGED)

# Camera intrinsic parameters (example values, should be replaced with actual values)
camera_intrinsics = [714.5459, 714.5459, 0, 0]  # Replace with actual intrinsic values

# Convert to point cloud
point_cloud = depth_to_pointcloud(depth_image, camera_intrinsics)

# Convert to Open3D PointCloud object for visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
