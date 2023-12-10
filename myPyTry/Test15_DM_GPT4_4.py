import numpy as np
import cv2
from tqdm import tqdm


def compute_depth_map(image1, image2, patch_size=5, max_iterations=10, threshold=0.1):
    # 随机初始化
    height, width, _ = image1.shape
    depth_map = np.random.rand(height, width)
    normals_map = np.random.rand(height, width, 3)

    for iteration in range(max_iterations):
        for y in range(height):
            for x in range(width):
                # 创建Patch
                patch1 = create_patch(image1, x, y, patch_size)

                # 最佳匹配初始化
                best_cost = float('inf')
                best_depth = depth_map[y, x]
                best_normal = normals_map[y, x]

                # 在第二张图像中搜索匹配的Patch
                for dy in range(-patch_size, patch_size + 1):
                    for dx in range(-patch_size, patch_size + 1):
                        x2 = x + dx
                        y2 = y + dy
                        if 0 <= x2 < width and 0 <= y2 < height:
                            patch2 = create_patch(image2, x2, y2, patch_size)
                            cost = np.mean((patch1 - patch2) ** 2)

                            # 更新最佳匹配
                            if cost < best_cost:
                                best_cost = cost
                                best_depth = depth_map[y2, x2]
                                best_normal = normals_map[y2, x2]

                # 更新深度图和法线图
                depth_map[y, x] = best_depth if best_cost < threshold else depth_map[y, x]
                normals_map[y, x] = best_normal
    print(iteration)

    return depth_map


def create_patch(image, x, y, size):
    height, width, _ = image.shape
    half_size = size // 2

    # 确保补丁不会超出图像边界
    startx = max(0, x - half_size)
    starty = max(0, y - half_size)
    endx = min(width, x + half_size + 1)
    endy = min(height, y + half_size + 1)

    # 提取补丁
    patch = image[starty:endy, startx:endx]

    # 如果补丁大小小于期望大小，则用0填充至正确大小
    if patch.shape[0] < size or patch.shape[1] < size:
        patch = np.pad(patch, ((0, size - patch.shape[0]), (0, size - patch.shape[1]), (0, 0)), mode='constant', constant_values=0)

    return patch


def display_depth_map(depth_map):
    # 将深度图标准化到0-255
    normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized_depth_map = np.uint8(normalized_depth_map)

    # 显示深度图
    cv2.imshow('Depth Map', normalized_depth_map)
    cv2.waitKey(0)  # 等待按键事件
    cv2.destroyAllWindows()



# 示例：加载两张图像
image1 = cv2.imread('Datasets/myCV/0000.jpg')
image2 = cv2.imread('Datasets/myCV/0001.jpg')

# 计算深度图
depth_map = compute_depth_map(image1, image2)
print( depth_map)
display_depth_map(depth_map)