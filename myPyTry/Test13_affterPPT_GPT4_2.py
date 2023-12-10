import os

import numpy as np
import cv2
from scipy.signal import convolve2d
import random
import open3d as o3d


def compute_ncc(patch1, patch2):
    """
    计算两个图像补丁之间的归一化互相关得分。
    """
    mean1, mean2 = np.mean(patch1), np.mean(patch2)
    std1, std2 = np.std(patch1), np.std(patch2)

    if std1 > 1e-6 and std2 > 1e-6:
        ncc = np.mean((patch1 - mean1) * (patch2 - mean2)) / (std1 * std2)
    else:
        ncc = 0

    return ncc


def patch_match_initialization(depth_map, normal_map):
    """
    随机初始化深度图和法线图
    """
    height, width = depth_map.shape
    # 随机初始化深度图和法线图
    random_depth_map = np.random.rand(height, width)
    random_normal_map = np.random.rand(height, width, 3)  # 法线有三个分量
    return random_depth_map, random_normal_map


def compute_homography_and_cost(ref_image, target_image, depth_map, normal_map, K, window_size=5):
    height, width = ref_image.shape
    cost_map = np.zeros((height, width))

    # 计算逆相机内参矩阵
    K_inv = np.linalg.inv(K)

    # 遍历图像的每个像素
    for y in range(window_size//2, height - window_size//2):
        for x in range(window_size//2, width - window_size//2):
            # 获取当前像素的深度和法线
            depth = depth_map[y, x]
            normal = normal_map[y, x]

            # 计算世界坐标系下的点
            point = depth * np.dot(K_inv, np.array([x, y, 1]))

            # 使用法线和深度信息来计算同态变换
            # 这里需要一个详细的同态变换计算过程
            # 例如，可以使用法线来计算旋转矩阵，然后结合深度信息进行变换

            # 提取参考图像中的补丁
            patch_ref = ref_image[y - window_size//2:y + window_size//2 + 1,
                                  x - window_size//2:x + window_size//2 + 1]

            # 假设目标补丁与参考补丁位置相同（需要替换为根据同态变换计算的实际位置）
            patch_target = target_image[y - window_size//2:y + window_size//2 + 1,
                                        x - window_size//2:x + window_size//2 + 1]

            # 计算NCC得分
            ncc_score = compute_ncc(patch_ref, patch_target)

            # 将NCC得分转换为成本
            cost_map[y, x] = 1 - ncc_score

    print("cost_map: ", cost_map)

    return cost_map

def spatial_propagation_and_random_assignment(depth_map, cost_map, iterations=5, window_size=3):
    height, width = depth_map.shape
    new_depth_map = depth_map.copy()

    for _ in range(iterations):
        # 遍历每个像素
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_cost = cost_map[y, x]
                current_depth = new_depth_map[y, x]

                # 空间传播：考虑相邻像素
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue

                        neighbor_x, neighbor_y = x + dx, y + dy
                        neighbor_depth = new_depth_map[neighbor_y, neighbor_x]
                        neighbor_cost = cost_map[neighbor_y, neighbor_x]

                        # 如果相邻像素具有更低的成本，则更新当前像素的深度
                        if neighbor_cost < current_cost:
                            current_depth = neighbor_depth
                            current_cost = neighbor_cost

                # 随机分配：在一定概率下随机更新当前像素的深度
                if random.random() < 0.1:  # 假设有10%的概率进行随机更新
                    current_depth = random.uniform(0, 1)  # 随机深度值

                new_depth_map[y, x] = current_depth

        # 可选：根据需要添加深度平滑或过滤步骤

    return new_depth_map

def apply_threshold(depth_map, cost_map, threshold=0.5):
    """
    应用阈值删除匹配质量低的点。

    :param depth_map: 深度图
    :param cost_map: 成本图
    :param threshold: 成本阈值
    :return: 更新后的深度图
    """
    # 复制深度图以避免直接修改原始数据
    filtered_depth_map = depth_map.copy()

    # 遍历成本图的每个像素
    height, width = cost_map.shape
    for y in range(height):
        for x in range(width):
            # 如果成本超过阈值，则删除（或标记为无效）该像素的深度
            if cost_map[y, x] > threshold:
                filtered_depth_map[y, x] = 0  # 你可以选择设置为0或其他标识无效深度的值

    return filtered_depth_map



def depth_map_refinement(depth_map, threshold=0.1, kernel_size=5):
    """
    完善深度图：去除噪声、平滑处理和填补空洞。

    :param depth_map: 原始深度图
    :param threshold: 用于边缘检测的深度阈值
    :param kernel_size: 滤波器的尺寸
    :return: 完善后的深度图
    """
    # 1. 去除离群点（基于深度差异）
    height, width = depth_map.shape
    refined_depth_map = depth_map.copy()
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # 获取当前像素及其周围像素的深度值
            depth_center = depth_map[y, x]
            depth_neighbors = depth_map[y - 1:y + 2, x - 1:x + 2]

            # 如果当前像素与周围像素的深度差异超过阈值，则认为是离群点
            if np.max(np.abs(depth_neighbors - depth_center)) > threshold:
                refined_depth_map[y, x] = np.median(depth_neighbors)

    # # 2. 应用中值滤波器进行平滑处理
    # refined_depth_map = cv2.medianBlur(refined_depth_map, kernel_size)

    # 3. 可选：填补空洞（例如，通过形态学运算）
    # 如果有必要，可以在这里添加进一步的空洞填补步骤

    return refined_depth_map


def merge_depth_maps(depth_maps, K, depth_diff_threshold=0.02):
    """
    合并多个深度图为一个统一的点云。

    :param depth_maps: 深度图的列表
    :param K: 相机内参矩阵
    :param depth_diff_threshold: 深度差异阈值，用于剔除不一致的映射
    :return: 合并后的点云
    """
    # 初始化点云
    merged_points = []

    # 遍历每个深度图
    for depth_map in depth_maps:
        height, width = depth_map.shape

        # 遍历深度图的每个像素
        for y in range(height):
            for x in range(width):
                depth = depth_map[y, x]

                # 跳过无效深度
                if depth <= 0:
                    continue

                # 将像素坐标和深度转换为世界坐标
                point = depth * np.dot(np.linalg.inv(K), np.array([x, y, 1]))
                merged_points.append(point)

    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(merged_points)

    # 可选：对点云进行进一步的过滤或处理
    # 例如，使用统计滤波器或半径滤波器去除离群点

    return point_cloud


K = np.array([[714.5459, 0, 0], [0, 714.5459, 0], [0, 0, 1]])  # 示例相机内参，f为焦距，cx，cy为主点坐标

image_list = []

for image in sorted(os.listdir('Datasets/myCV')):
    if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
        image_list.append('Datasets/myCV' + '\\' + image)


def generate_depth_map(ref_image, target_image, camera_matrix, baseline=0.01, num_disparities=16, block_size=15):
    """
    使用相机内参和基线距离生成参考图像和目标图像之间的深度图。

    :param ref_image: 参考图像
    :param target_image: 目标图像
    :param camera_matrix: 相机内参矩阵
    :param baseline: 相机之间的基线距离
    :param num_disparities: 视差搜索范围
    :param block_size: 匹配块的大小
    :return: 生成的深度图和可视化深度图
    """
    # 转换图像为灰度（如果它们不是灰度图）
    if len(ref_image.shape) > 2:
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    if len(target_image.shape) > 2:
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 创建立体匹配对象
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    # 计算视差图
    disparity = stereo.compute(ref_image, target_image).astype(np.float32)

    # 视差图后处理，去除异常值
    min_disparity = disparity.min()
    max_disparity = disparity.max()
    disparity = np.where((disparity > min_disparity) & (disparity <= max_disparity), disparity, 0)

    # 视差图转换为深度图
    focal_length = camera_matrix[0, 0]  # 相机焦距
    # 避免除以零
    disparity[disparity == 0] = 0.1
    depth_map = (focal_length * baseline) / disparity

    # 深度图归一化，以便于显示
    depth_map_display = cv2.normalize(depth_map, None, alpha=10, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_map_display = np.uint8(depth_map_display)

    return depth_map

def process_image_pair(ref_image_path, target_image_path):
    # 假设你已经有了处理图像对以生成深度图的函数
    # 此处是那些函数的简化示例，具体实现需要你根据自己的需求来填写

    # 加载图像
    ref_image = cv2.imread(ref_image_path, 0)
    target_image = cv2.imread(target_image_path, 0)

    depth_map = generate_depth_map(ref_image, target_image,K)

    # 返回生成的深度图
    return depth_map


depth_maps = []
# 迭代处理图像
for i in range(len(image_list) - 1):
    ref_image_path = image_list[i]
    target_image_path = image_list[i + 1]

    # 处理图像对并生成深度图
    depth_map = process_image_pair(ref_image_path, target_image_path)

    # 将生成的深度图添加到列表中
    depth_maps.append(depth_map)


def generate_normal_map(ref_image_path, target_image_path):
    ref_image = cv2.imread(ref_image_path, 0)
    target_image = cv2.imread(target_image_path, 0)
    depth_map = depth_maps[i]

    # 对深度图应用高斯模糊
    blurred_depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

    # 使用Sobel算子计算梯度
    sobelx = cv2.Sobel(blurred_depth_map, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred_depth_map, cv2.CV_64F, 0, 1, ksize=5)

    # 初始化法线图
    height, width = blurred_depth_map.shape
    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    # 计算法线向量
    normal_map[..., 0] = -sobelx  # x分量
    normal_map[..., 1] = -sobely  # y分量
    normal_map[..., 2] = 1.0      # z分量

    # 归一化法线向量
    norm = np.linalg.norm(normal_map, axis=2)
    norm[norm == 0] = 1  # 避免除以零
    normal_map /= np.dstack((norm, norm, norm))

    return normal_map


normal_maps = []
# 迭代处理图像
for i in range(len(image_list) - 1):
    ref_image_path = image_list[i]
    target_image_path = image_list[i + 1]

    # 处理图像对并生成深度图
    normal_map = generate_normal_map(ref_image_path, target_image_path)

    # 将生成的深度图添加到列表中
    normal_maps.append(normal_map)


# 迭代处理图像
for i in range(len(image_list) - 1):
    # 加载参考图像和目标图像
    ref_image = cv2.imread(image_list[i], 0)
    target_image = cv2.imread(image_list[i+1], 0)

    # 假设你已经有了初始的深度图和法线图
    initial_depth_map = depth_maps[i] # 示例深度图
    initial_normal_map = normal_maps[i] # 示例法线图

    cost_map = compute_homography_and_cost(ref_image, target_image, depth_map, normal_map, K)
    depth_map = spatial_propagation_and_random_assignment(depth_map, cost_map)
    depth_map = apply_threshold(depth_map, cost_map)
    refined_depth_map = depth_map_refinement(depth_map)

    # 将完善后的深度图添加到列表中
    depth_maps.append(refined_depth_map)

# 合并深度图
merged_point_cloud = merge_depth_maps(depth_maps, K)

# 可视化合并后的点云
o3d.visualization.draw_geometries([merged_point_cloud])