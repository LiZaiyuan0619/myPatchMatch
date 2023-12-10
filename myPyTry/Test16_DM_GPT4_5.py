import numpy as np
import cv2

import numpy as np
import cv2

def generate_depth_map(ref_image, target_image, camera_matrix, baseline, num_disparities=16, block_size=15):
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

    depth_map_display = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # depth_map_display = np.uint8(depth_map_display)

    return depth_map, depth_map_display


# 使用示例
camera_matrix = np.array([[714.5459, 0, 0], [0, 714.5459, 0], [0, 0, 1]])
baseline = 0.1  # 示例基线距离
ref_image = cv2.imread('Datasets/myCV/0000.jpg')
target_image = cv2.imread('Datasets/myCV/0001.jpg')
depth_map, depth_map_display = generate_depth_map(ref_image, target_image, camera_matrix, baseline)
cv2.imwrite('Test16_depth_map2.png', depth_map)
# 显示深度图
cv2.imshow('Depth Map', depth_map_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
