import pandas as pd
import numpy as np
import open3d as o3d
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
data_path = "res/myCV_pose_array.csv"
w = 480
h = 640
data = pd.read_csv(data_path, header=None)
points = np.zeros((w*h, 3), dtype=np.float32)
n=0
for i in range(w):  # 将原来的高度变量h改为宽度变量w
    for j in range(h):  # 将原来的宽度变量w改为高度变量h
        deep = data.iloc[j, i]  # 交换i和j的位置
        points[n][0] = i  # 交换i和j的位置
        points[n][1] = j  # 交换i和j的位置
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