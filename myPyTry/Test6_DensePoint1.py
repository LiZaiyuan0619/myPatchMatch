# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# imgL = cv.imread('Datasets/myCV/0000.jpg',0)
# imgR = cv.imread('Datasets/myCV/0001.jpg',0)
#
# stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity,'gray')
# plt.show()
###############################################################
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# 文件夹路径
dataset_path = 'Datasets/myCV/'

# 获取文件夹下所有图像文件
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

# 读取第一张图像，作为基准图像
base_img = cv.imread(os.path.join(dataset_path, image_files[0]), 0)

# 创建StereoSGBM对象
stereo = cv.StereoSGBM_create(numDisparities=16, blockSize=15)

# 初始化深度图
depth_map = np.zeros_like(base_img, dtype=np.float32)

# 循环处理每一张图像
for image_file in image_files[1:]:
    # 读取当前图像
    current_img = cv.imread(os.path.join(dataset_path, image_file), 0)

    # 计算视差图
    disparity = stereo.compute(base_img, current_img)

    # 更新深度图
    depth_map += disparity

# 显示深度图
plt.imshow(depth_map, 'jet')  # 使用'jet' colormap，可以根据需要选择其他colormap
plt.title("Depth Map")
plt.show()
# 保存深度图
cv.imwrite('depth_map.png', depth_map)
################################################################
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # 文件夹路径
# dataset_path = 'Datasets/myCV/'
#
# # 获取文件夹下所有图像文件
# image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
#
# # 读取第一张图像，作为基准图像
# base_img = cv.imread(os.path.join(dataset_path, image_files[0]), 0)
#
# # 创建StereoSGBM对象
# stereo = cv.StereoSGBM_create(numDisparities=16, blockSize=15)
#
# # 初始化深度图
# depth_map = np.zeros_like(base_img, dtype=np.float32)
#
# # 循环处理每一张图像
# for image_file in image_files[1:]:
#     # 读取当前图像
#     current_img = cv.imread(os.path.join(dataset_path, image_file), 0)
#
#     # 计算视差图
#     disparity = stereo.compute(base_img, current_img)
#
#     # 更新深度图
#     depth_map += disparity
#
# # 映射深度图到伪彩色
# depth_map_color = cv.applyColorMap((depth_map/depth_map.max() * 255).astype(np.uint8), cv.COLORMAP_JET)
#
# # 显示深度图
# plt.imshow(depth_map_color)
# plt.title("Colored Depth Map")
# plt.show()
#
# # 保存深度图
# cv.imwrite('depth_map_color.png', depth_map_color)
########################################################################
# import cv2 as cv
# import numpy as np
# import open3d as o3d
# import os
#
# # 文件夹路径
# dataset_path = 'Datasets/myCV/'
#
# # 获取文件夹下所有图像文件
# image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
#
# # 读取第一张图像，作为基准图像
# base_img = cv.imread(os.path.join(dataset_path, image_files[0]), 0)
#
# # 创建StereoSGBM对象
# stereo = cv.StereoSGBM_create(numDisparities=16, blockSize=15)
#
# # 初始化深度图
# depth_map = np.zeros_like(base_img, dtype=np.float32)
#
# # 循环处理每一张图像
# for image_file in image_files[1:]:
#     # 读取当前图像
#     current_img = cv.imread(os.path.join(dataset_path, image_file), 0)
#
#     # 计算视差图
#     disparity = stereo.compute(base_img, current_img)
#
#     # 更新深度图
#     depth_map += disparity
#
# # 构建点云
# fx = 714.5459
# fy = 714.5459
# cx = 0
# cy = 0
#
# focal_length = fx  # 使用 fx 或 fy 均可
# intrinsics = o3d.camera.PinholeCameraIntrinsic(
#     width=base_img.shape[1],
#     height=base_img.shape[0],
#     fx=fx,
#     fy=fy,
#     cx=cx,
#     cy=cy
# )
#
# # 根据深度图生成稠密点云
# h, w = depth_map.shape[:2]
# u, v = np.meshgrid(np.arange(w), np.arange(h))
# z = depth_map / 1000.0  # Assuming depth is in millimeters, convert to meters
# x = (u - cx) * z / fx
# y = (v - cy) * z / fy
# points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
#
# # Create PointCloud
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)
#
# # Assign colors based on the intensity values from the base image
# colors = cv.cvtColor(base_img, cv.COLOR_GRAY2BGR)
# colors = colors.reshape(-1, 3) / 255.0
# point_cloud.colors = o3d.utility.Vector3dVector(colors)
#
# # Save PointCloud
# o3d.io.write_point_cloud("dense_point_cloud.ply", point_cloud)
# 生成的PLY文件看不出来是我的图像
