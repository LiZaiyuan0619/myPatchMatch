import cv2
import numpy as np

def compute_normal_map(depth_map):
    # 计算深度图的梯度
    dx, dy = np.gradient(depth_map)

    # 初始化法线图
    normal_map = np.zeros(depth_map.shape + (3,), dtype=np.float32)

    # 计算法线的x, y, z分量
    # 注意这里假设深度图是沿z轴变化的
    normal_map[..., 0] = -dx  # x分量
    normal_map[..., 1] = -dy  # y分量
    normal_map[..., 2] = 1.0  # z分量

    # 归一化法线向量
    norm = np.linalg.norm(normal_map, axis=2)
    normal_map /= np.dstack((norm, norm, norm))

    return normal_map

# 假设你已经有了一个深度图
depth_map = cv2.imread("depth_map.png", cv2.IMREAD_UNCHANGED)

# 计算法线图
normal_map = compute_normal_map(depth_map)

# 保存或使用法线图
cv2.imwrite("my_normal_map.png", normal_map * 255)
