import numpy as np
import cv2
from scipy.signal import convolve2d
from matplotlib import pyplot as plt


def compute_homography(depth, normal, K_ref, K_target, R, T):
    """
    Compute the homography matrix for a patch given the depth, normal, camera intrinsics,
    and the relative pose between reference and target cameras.

    Parameters:
    - depth: The depth value at the pixel.
    - normal: The normal vector at the pixel.
    - K_ref: The camera intrinsic matrix for the reference camera.
    - K_target: The camera intrinsic matrix for the target camera.
    - R: The rotation matrix from reference to target camera.
    - T: The translation vector from reference to target camera.

    Returns:
    - H: The homography matrix.
    """
    # Create the point in homogenous coordinates
    point = np.array([depth, depth, depth])

    # Compute the plane normal in reference camera space
    normal_cam_space = np.linalg.inv(K_ref).dot(normal)

    # Compute the homography components
    d = np.dot(normal_cam_space, point)
    H = K_target.dot(R - np.outer(T, normal_cam_space) / d).dot(np.linalg.inv(K_ref))

    return H


def compute_homography_and_cost(reference_image, target_image, depth_map, normal_map, patch_size=7):
    """
    Compute the homography and cost for patches in the reference image against the target image.

    Parameters:
    - reference_image: The reference image.
    - target_image: The target image.
    - depth_map: The depth map for the reference image.
    - normal_map: The normal map for the reference image.
    - patch_size: The size of the patch to consider for each pixel.

    Returns:
    - cost_map: A cost map where each value represents the cost of the patch centered at the corresponding pixel.
    """
    height, width = reference_image.shape[:2]
    cost_map = np.zeros((height, width))

    # Define the patch radius
    patch_radius = patch_size // 2

    # Precompute the patch template for NCC
    patch_template = np.ones((patch_size, patch_size))

    # Iterate over all pixels in the reference image
    for y in range(patch_radius, height - patch_radius):
        for x in range(patch_radius, width - patch_radius):
            # Get the current patch from the reference image
            patch_ref = reference_image[y - patch_radius:y + patch_radius + 1,
                                        x - patch_radius:x + patch_radius + 1]



            # Compute the homography for the current patch
            # This is a simplified version: in practice, you would compute the homography based on the depth and normal.
            # Here we'll just simulate a homography with a translation for demonstration purposes.
            h_translation = np.array([[1, 0, depth_map[y, x]],
                                      [0, 1, normal_map[y, x, 2]],
                                      [0, 0, 1]])

            # Apply the homography to get the corresponding patch in the target image
            patch_target = cv2.warpPerspective(target_image, h_translation, (width, height))[y - patch_radius:y + patch_radius + 1,
                                                                                              x - patch_radius:x + patch_radius + 1]

            # Compute the Normalized Cross Correlation (NCC) cost
            mean_ref = np.mean(patch_ref)
            mean_target = np.mean(patch_target)
            std_ref = np.std(patch_ref)
            std_target = np.std(patch_target)

            if std_ref > 0 and std_target > 0:
                ncc = np.mean((patch_ref - mean_ref) * (patch_target - mean_target)) / (std_ref * std_target)
                cost = 1 - ncc  # Cost is inversely related to NCC
            else:
                cost = 1  # Max cost if there's no variation in intensity

            cost_map[y, x] = cost

    return cost_map

import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

def random_initialization(reference_image, target_image, depth_range=(0.1, 10.0)):
    """
    Initialize the depth map and normal map randomly for the reference image.

    Parameters:
    - reference_image: The reference image.
    - target_image: The target image (not used in this function, but typically used in subsequent steps).
    - depth_range: The range of possible depth values.

    Returns:
    - depth_map: A randomly initialized depth map.
    - normal_map: A randomly initialized normal map.
    """
    # Get the dimensions of the reference image
    height, width = reference_image.shape[:2]

    # Initialize the depth map with random values within the specified range
    depth_map = np.random.uniform(depth_range[0], depth_range[1], (height, width))

    # Initialize the normal map with random unit vectors
    # For each pixel, a 3D normal vector with values between -1 and 1 is generated and then normalized
    normal_map = np.random.uniform(-1, 1, (height, width, 3))
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map /= norm  # Normalize the vectors to have a length of 1

    return depth_map, normal_map


########################################################################################################################


K = np.array([[714.5459, 0, 0], [0, 714.5459, 0], [0, 0, 1]])  # 示例相机内参，f为焦距，cx，cy为主点坐标

image_list = []

for image in sorted(os.listdir('Datasets/myCV')):
    if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
        image_list.append('Datasets/myCV' + '\\' + image)

# Load images using OpenCV
ref_image = cv2.imread(image_list[0], cv2.IMREAD_COLOR)  # replace with your image path
target_image = cv2.imread(image_list[1], cv2.IMREAD_COLOR)  # replace with your image path

# Perform random initialization
depth_map, normal_map = random_initialization(ref_image, target_image)

# Compute the homography and cost
cost_map = compute_homography_and_cost(ref_image, target_image, depth_map, normal_map)

# Visualize the cost map
plt.figure(figsize=(6, 6))
plt.title('Cost Map')
plt.imshow(cost_map, cmap='jet')
plt.colorbar()
plt.show()
