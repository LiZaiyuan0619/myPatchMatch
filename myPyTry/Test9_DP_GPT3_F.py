import open3d as o3d
import numpy as np

def generate_dense_point_cloud(sparse_pcd_file, depth_map_file, camera_intrinsics):
    # 读取稀疏点云和深度图
    sparse_pcd = o3d.io.read_point_cloud(sparse_pcd_file)
    depth_map = o3d.io.read_image(depth_map_file)

    # 获取深度图的像素值
    depth_image = np.asarray(depth_map)
    depths = depth_image / 1000.0  # 将深度值转换为米

    # 使用稀疏点云和深度图生成稠密点云
    dense_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=o3d.Image(depths),
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=depth_map.width, height=depth_map.height, fx=camera_intrinsics[0], fy=camera_intrinsics[1],
            cx=camera_intrinsics[2], cy=camera_intrinsics[3]
        ),
        extrinsic=np.identity(4)
    )

    # 将稠密点云与稀疏点云合并
    dense_pcd += sparse_pcd

    return dense_pcd


# 示例使用
sparse_pcd_file = "res/myCV_pointcloud.pcd.pcd"
depth_map_file = "depth_map_color.png"
camera_intrinsics = [714.5459, 714.5459, 0, 0]  # 请替换为实际相机参数

dense_pcd = generate_dense_point_cloud(sparse_pcd_file, depth_map_file, camera_intrinsics)

# 保存稠密点云
o3d.io.write_point_cloud("temp_pcd.pcd", dense_pcd)
