import pandas as pd
import numpy as np
import open3d as o3d
from tqdm import tqdm
#ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud("res/myCV.ply")
#pcd = o3d.io.read_point_cloud("./my_points.txt", format='xyz')
#pcd = o3d.io.read_point_cloud("./face.pcd")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
camera_factor = 1;
camera_fx = 714.5459;
camera_fy = 714.5459;

camera_cx = 0;
camera_cy = 0;

data_path = "depth_map_csv.csv"

data = pd.read_csv(data_path)

# 获取图像的宽度和高度
w = data['column'].max() + 1  # 因为列是从0开始的，所以要加1
h = data['row'].max() + 1  # 因为行是从0开始的，所以要加1

points = np.zeros((w * h, 3), dtype=np.float32)
n = 0

for i in tqdm(range(h)):
    for j in range(w):
        # 通过查询data DataFrame获取深度值
        depth_data = data[(data['row'] == i) & (data['column'] == j)]['depth']

        # 检查是否找到了深度值，如果找到了就赋值，否则默认为0
        if not depth_data.empty:
            deep = float(depth_data.values[0])
        else:
            deep = 0.0

        # 存储坐标和深度信息
        points[n][0] = (j-camera_cx)*points[n][2]/camera_fx
        points[n][1] = (i-camera_cy)*points[n][2]/camera_fy
        points[n][2] = deep
        n = n + 1

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
#o3d.io.write_point_cloud("../../test_data/sync.ply", pcd)

print("==========")
print(pcd)
print(np.asarray(pcd.points))
print("==========")




o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])