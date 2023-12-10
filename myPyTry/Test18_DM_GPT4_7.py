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




import numpy as np
import cv2
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



def compute_homography_and_cost(ref_image, target_image, depth_map, normal_map, K_ref, K_target, R, T, patch_size=7):
    """
    Compute the homography and cost for patches in the reference image against the target image.

    Parameters:
    - ref_image: The reference image.
    - target_image: The target image.
    - depth_map: The depth map for the reference image.
    - normal_map: The normal map for the reference image.
    - K_ref: Intrinsic matrix of the reference camera.
    - K_target: Intrinsic matrix of the target camera.
    - R: Rotation matrix from reference to target camera.
    - T: Translation vector from reference to target camera.
    - patch_size: The size of the patch to consider for each pixel.

    Returns:
    - cost_map: A cost map where each value represents the cost of the patch centered at the corresponding pixel.
    """
    height, width = ref_image.shape[:2]
    cost_map = np.zeros((height, width))

    # Define the patch radius
    patch_radius = patch_size // 2

    for y in range(patch_radius, height - patch_radius):
        for x in range(patch_radius, width - patch_radius):
            # Calculate Homography for current pixel
            depth = depth_map[y, x]
            normal = normal_map[y, x]
            H = compute_homography(depth, normal, K_ref, K_target, R, T)

            # Warp the patch from reference to target image
            src_pts = np.float32([[x - patch_radius, y - patch_radius],
                                  [x + patch_radius, y - patch_radius],
                                  [x + patch_radius, y + patch_radius],
                                  [x - patch_radius, y + patch_radius]]).reshape(-1,1,2)
            dst_pts = cv2.perspectiveTransform(src_pts, H)
            [min_x, min_y] = np.int32(dst_pts.min(axis=0).ravel() - 0.5)
            [max_x, max_y] = np.int32(dst_pts.max(axis=0).ravel() + 0.5)
            warp_size = (max_x - min_x, max_y - min_y)

            ref_patch = ref_image[y - patch_radius:y + patch_radius + 1, x - patch_radius:x + patch_radius + 1]
            patch_target = cv2.warpPerspective(ref_patch, H, warp_size)

            # Compute NCC cost
            mean_ref = np.mean(ref_patch)
            mean_target = np.mean(patch_target)
            std_ref = np.std(ref_patch)
            std_target = np.std(patch_target)

            if std_ref > 0 and std_target > 0:
                ncc = np.mean((ref_patch - mean_ref) * (patch_target - mean_target)) / (std_ref * std_target)
                cost = 1 - ncc  # Cost is inversely related to NCC
            else:
                cost = 1  # Max cost if there's no variation in intensity

            cost_map[y, x] = cost

    return cost_map

K = np.array([[714.5459, 0, 0], [0, 714.5459, 0], [0, 0, 1]])  # 示例相机内参，f为焦距，cx，cy为主点坐标

image_list = []

for image in sorted(os.listdir('Datasets/myCV')):
    if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
        image_list.append('Datasets/myCV' + '\\' + image)

# Load images using OpenCV
ref_image = cv2.imread(image_list[0], cv2.IMREAD_COLOR)  # replace with your image path
target_image = cv2.imread(image_list[1], cv2.IMREAD_COLOR)  # replace with your image path

# R = np.eye(3)
# T = np.zeros(3)
R = np.array([[0.99590467, -0.01363085, 0.08937612],
              [0.01685576, 0.99923009, -0.03542757],
              [-0.0888244, 0.03678899, 0.99536767]])

T = np.array([-0.99900655, -0.01930007, -0.0401673])


# Perform random initialization
depth_map, normal_map = random_initialization(ref_image, target_image)

# Compute the homography and cost
cost_map = compute_homography_and_cost(ref_image, target_image, depth_map, normal_map, K, K, R, T)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Random Depth Map')
plt.imshow(depth_map, cmap='jet')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Random Normal Map')
# Display the normal map by normalizing the values to be between 0 and 1
plt.imshow((normal_map + 1) / 2)
plt.show()


# Visualize the cost map
plt.figure(figsize=(6, 6))
plt.title('Cost Map')
plt.imshow(cost_map, cmap='jet')
plt.colorbar()
plt.show()


def compute_ncc_cost(patch_a, patch_b):
    """
    Compute the NCC cost between two patches.

    Parameters:
    - patch_a: Patch from the first image.
    - patch_b: Corresponding patch from the second image.

    Returns:
    - ncc_cost: The NCC cost between the two patches.
    """
    mean_a = np.mean(patch_a)
    mean_b = np.mean(patch_b)
    std_a = np.std(patch_a)
    std_b = np.std(patch_b)

    if std_a > 0 and std_b > 0:
        ncc = np.sum((patch_a - mean_a) * (patch_b - mean_b)) / (std_a * std_b * patch_a.size)
        return 1 - ncc  # Lower cost indicates better match
    else:
        return 1  # Max cost for no variation


def spatial_propagation_and_random_assignment(ref_image, target_image, depth_map, normal_map, cost_map, patch_size=7, iterations=5, max_depth_change=0.1, max_normal_change=0.1):
    """
    Update depth and normal maps based on spatial propagation and random assignment using precomputed cost map.

    Parameters:
    - ref_image: Reference image.
    - target_image: Target image.
    - depth_map: Initial depth map.
    - normal_map: Initial normal map.
    - cost_map: Precomputed cost map.
    - patch_size: Size of the patch.
    - iterations: Number of iterations to run the algorithm.
    - max_depth_change: Maximum change allowed in depth in random assignment.
    - max_normal_change: Maximum change allowed in normal vector in random assignment.

    Returns:
    - Updated depth_map and normal_map.
    """
    height, width = ref_image.shape[:2]

    for iteration in range(iterations):
        for y in range(height):
            for x in range(width):
                current_cost = cost_map[y, x]

                # Spatial propagation: check top and left neighbors
                for dy, dx in [(-1, 0), (0, -1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_cost = cost_map[ny, nx]
                        if neighbor_cost < current_cost:
                            # Update depth, normal and cost for the current pixel
                            depth_map[y, x] = depth_map[ny, nx]
                            normal_map[y, x] = normal_map[ny, nx]
                            current_cost = neighbor_cost

                # Random assignment
                # Randomly perturb depth and normal
                new_depth = depth_map[y, x] + np.random.uniform(-max_depth_change, max_depth_change)
                new_normal = normal_map[y, x] + np.random.uniform(-max_normal_change, max_normal_change)
                new_normal /= np.linalg.norm(new_normal)  # Normalize the normal vector

                H = compute_homography(new_depth, new_normal, K, K, R, T)

                # 应用warpPerspective到整个图像
                warped_image = cv2.warpPerspective(ref_image, H, (width, height))

                patch_size_half = patch_size // 2
                y_min = max(y - patch_size_half, 0)
                y_max = min(y + patch_size_half + 1, height)
                x_min = max(x - patch_size_half, 0)
                x_max = min(x + patch_size_half + 1, width)

                # 从参考图像和变换后的图像中提取对应的patch
                ref_patch = ref_image[y_min:y_max, x_min:x_max]
                patch_target = warped_image[y_min:y_max, x_min:x_max]

                # 确保patch_target不为空
                if patch_target.size == 0:
                    continue

                # 计算NCC成本
                new_cost = compute_ncc_cost(ref_patch, patch_target)

                if new_cost < current_cost:
                    depth_map[y, x] = new_depth
                    normal_map[y, x] = new_normal
                    cost_map[y, x] = new_cost

    return depth_map, normal_map

depth_map, normal_map = spatial_propagation_and_random_assignment(ref_image, target_image, depth_map, normal_map, cost_map, patch_size=7, iterations=10, max_depth_change=0.1, max_normal_change=0.1)

# 保存深度图
# 深度图可能需要归一化以便于观察
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite('depth_map125.png', depth_map_normalized)

# 保存法线图
# 法线图的每个通道应该在0到255之间，因此需要转换和缩放
normal_map_normalized = ((normal_map + 1) / 2 * 255).astype('uint8')
cv2.imwrite('normal_map125.png', normal_map_normalized)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Depth Map')
plt.imshow(depth_map, cmap='jet')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Normal Map')
# Display the normal map by normalizing the values to be between 0 and 1
plt.imshow((normal_map + 1) / 2)
plt.show()