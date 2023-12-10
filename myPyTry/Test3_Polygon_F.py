import numpy as np
import open3d as o3d

import open3d as o3d
import numpy as np
import numpy as np
#pyvista 用于处理三维数据和可视化
import pyvista as pv
from sklearn.neighbors import KDTree
import os
from scipy.optimize import least_squares
# from tomlkit import boolean
from tqdm import tqdm
import open3d as o3d
import cv2
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


# def create_mesh(pcd, normals, radius=0.1):
#     # 为高效的邻近点搜索创建 KDTree
#     kdtree = o3d.geometry.KDTreeFlann(pcd)
#
#     # 创建一个空的三角网格
#     mesh = o3d.geometry.TriangleMesh()
#
#     # 遍历每个点
#     for i in range(len(pcd.points)):
#         point = pcd.points[i]
#         normal = normals[i]
#
#         # 在指定半径内查找邻近点
#         k, _ = kdtree.search_radius_vector_3d(point, radius)
#         neighbors = np.asarray(k)
#
#         # 为每个点及其邻近点创建三角网格
#         if len(neighbors) >= 3:
#             triangles = []
#             for j in range(1, len(neighbors) - 1):
#                 triangles.append([i, neighbors[j], neighbors[j + 1]])
#
#             # 将三角形添加到网格中
#             mesh.vertices += np.asarray(pcd.points)[neighbors]
#             mesh.triangles += triangles
#
#             # 为每个顶点设置法向量
#             mesh.vertex_normals += [normal] * len(neighbors)
#
#     return mesh

def save_sfm_result_to_pcd(total_points, total_colors, output_pcd_file):
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()

    # 设置点云的坐标和颜色
    pcd.points = o3d.utility.Vector3dVector(total_points)
    pcd.colors = o3d.utility.Vector3dVector(total_colors / 255.0)  # 颜色值通常在 [0, 255] 范围内，需要归一化到 [0, 1]

    # 保存为PCD文件
    o3d.io.write_point_cloud(output_pcd_file, pcd)

# 定义一个Image_loader类，用于加载图像
# 我们主要做了以下的工作：读取了相机内参，加载了图像列表，执行了下采样。
class Image_loader():
    def __init__(self, img_dir: str, downscale_factor: float):
        # loading the Camera intrinsic parameters K
        # 加载相机内参数K
        with open(img_dir + '\\K.txt') as f:
            self.K = np.array(
                list((map(lambda x: list(map(lambda x: float(x), x.strip().split(' '))), f.read().split('\n')))))
            self.image_list = []
            print(self.K)
        # Loading the set of images
        # 加载图像列表
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
                self.image_list.append(img_dir + '\\' + image)

        # 获取当前工作目录
        self.path = os.getcwd()
        # 设置下采样因子
        self.factor = downscale_factor
        # 执行下采样
        self.downscale()

    # self.K 是相机矩阵，通常是一个3x3的矩阵，其中包含了相机的内部参数。
    # self.factor 是一个下采样因子，用于指定下采样的程度。这个因子可以是一个整数或浮点数。
    def downscale(self) -> None:
        '''
        Downscales the Image intrinsic parameter acc to the downscale factor
        '''
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor

    def downscale_image(self, image):
        # for _ in range(1,int(self.factor / 2) + 1):
        #     image = cv2.pyrDown(image)
        return image


class Sfm():
    def __init__(self, img_dir: str, downscale_factor: float = 2.0) -> None:
        '''
            Initialise and Sfm object.
            img_dir: directory of the images
        '''
        self.img_obj = Image_loader(img_dir, downscale_factor)

    def triangulation(self, point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2) -> tuple:
        '''
        据二维矢量和投影矩阵对三维点进行三角剖分
        返回第一台相机的投影矩阵、第二台相机的投影矩阵、点云
        该函数用于查找两幅图像的公共点
        '''
        # point_2d_1 和 point_2d_2 是两个视角下的对应点的二维坐标，通常是两个相机中捕捉到的同一物体上的同一个点的对应位置。
        # projection_matrix_1.T 和 projection_matrix_2.T 是两个相机的投影矩阵的转置。这些投影矩阵包含了相机的内部参数和外部参数。
        pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
        # projection_matrix_1.T 和 projection_matrix_2.T 分别返回了投影矩阵的转置。这些矩阵通常包含了相机的内部和外部参数
        # (pt_cloud / pt_cloud[3]) 将三维点云中的坐标除以其齐次坐标的尺度因子，以获得真实的三维坐标。
        # 这一步是为了将齐次坐标转换为普通的三维坐标。返回的结果包含了两个投影矩阵和三维点云的坐标。
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])

    def PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
        '''
        使用 RANSAC 方案从 3D-2D 点对应关系中查找物体姿态。
        返回 旋转矩阵、平移矩阵、图像点、物体点、旋转向量
        此函数用于查找
        '''
        if initial == 1:
            obj_point = obj_point[:, 0, :]
            image_point = image_point.T
            rot_vector = rot_vector.T
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff,
                                                                     cv2.SOLVEPNP_ITERATIVE)
        # Converts a rotation matrix to a rotation vector or vice versa
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

    def reprojection_error(self, obj_points, image_points, transform_matrix, K, homogenity) -> tuple:
        '''
        计算重投影误差，即投影点与实际点之间的距离。
        返回总误差、对象点数
        '''
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc,
                               np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        '''
        计算捆绑调整过程中的重投影误差
        返回误差
        '''
        transform_matrix = obj_points[0:12].reshape((3, 4))
        K = obj_points[12:21].reshape((3, 3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest / 2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:]) / 3), 3))
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        error = [(p[idx] - image_points[idx]) ** 2 for idx in range(len(p))]
        return np.array(error).ravel() / len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        '''
        对图像和对象点进行BA
        返回对象点、图像点、变换矩阵
        '''
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

        values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol=r_error).x
        K = values_corrected[12:21].reshape((3, 3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:]) / 3), 3)), values_corrected[
                                                                                                      21:21 + rest].reshape(
            (2, int(rest / 2))).T, values_corrected[0:12].reshape((3, 4))

    def to_ply(self, path, point_cloud, colors) -> None:
        '''
        Generates the .ply which can be used to open the point cloud
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')

    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        '''
        查找图像 1 和 2、图像 2 和 3 之间的公共点
        返回图像 1-2 的公共点、图像 2-3 的公共点、公共点 1-2 的掩码、公共点 2-3 的掩码
        '''
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2

    def find_features(self, image_0, image_1) -> tuple:
        '''
        使用筛分算法和 KNN 进行特征检测
        返回图像 1 和图像 2 的关键点（特征
        '''

        sift = cv2.xfeatures2d.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32(
            [key_points_1[m.trainIdx].pt for m in feature])

    def __call__(self, enable_bundle_adjustment=1):
        # 创建名为 'image' 的窗口，可以用于显示图像
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # 将相机内部参数（K矩阵）展平为一个一维数组，并存储在 pose_array 中。
        pose_array = self.img_obj.K.ravel()
        print("Pose Array: ", pose_array)
        # 表示第一个图像的初始变换矩阵，为一个单位矩阵
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        # 是一个尚未初始化的矩阵，稍后会被用于存储第二个图像的变换矩阵。
        transform_matrix_1 = np.empty((3, 4))

        # 这里使用相机内部参数和变换矩阵计算第一个图像的相机姿态矩阵 pose_0。第二个图像的相机姿态矩阵 pose_1 被初始化为空。
        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        print("Pose 0: ", pose_0)
        pose_1 = np.empty((3, 4))
        print("Pose 1: ", pose_1)
        # 初始化用于存储三维点云和对应颜色的数组
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        # 通过 cv2.imread 读取两幅图像，然后使用 self.img_obj.downscale_image 函数对图像进行下采样
        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))
        if image_0 is None or image_1 is None:
            print("Error: Could not read image")
            return

        # 使用 self.find_features 函数找到两幅图像的特征点并进行匹配。
        feature_0, feature_1 = self.find_features(image_0, image_1)
        print("First Feature 0: ", feature_0)

        # Essential matrix
        # 使用 cv2.findEssentialMat 估计本质矩阵，并根据 RANSAC 算法剔除外点。然后，根据掩码 em_mask 筛选特征点
        essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, self.img_obj.K, method=cv2.RANSAC,
                                                         prob=0.999, threshold=0.4, mask=None)
        print("First Essential Matrix: ", essential_matrix)
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        # cv2.recoverPose 函数从本质矩阵中恢复相机的旋转矩阵 rot_matrix 和平移矩阵 tran_matrix。
        # 同时，返回一个二值掩码 em_mask，标识哪些点被用于估计相机运动。
        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.img_obj.K)
        print(" First Rot Matrix: ", rot_matrix)
        # 使用 em_mask 掩码筛选出用于估计相机运动的特征点
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]
        print(" First Feature 0: ", feature_0)
        # 使用相机运动的旋转矩阵和平移矩阵来更新第二个图像的变换矩阵 transform_matrix_1。
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3],
                                                                          tran_matrix.ravel())
        print("First Transform Matrix 1: ", transform_matrix_1)

        # 使用相机内部参数 self.img_obj.K 和更新后的变换矩阵 transform_matrix_1 计算第二个图像的相机姿态矩阵。
        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)

        # 使用自定义的 self.triangulation 函数进行三角化，得到三维点云坐标 points_3d。
        feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
        # 使用自定义的 self.reprojection_error 函数计算重投影误差，这是衡量三维点云投影到图像上与实际特征点的误差。
        error, points_3d = self.reprojection_error(points_3d, feature_1, transform_matrix_1, self.img_obj.K,
                                                   homogenity=1)
        print("First Error: ", error)
        print("First Points 3D: ", points_3d)
        # ideally error < 1
        # print("REPROJECTION ERROR: ", error)
        # 使用自定义的 self.PnP 函数进行透视-非透视（Perspective-n-Point，PnP）求解，得到更准确的相机位姿。
        _, _, feature_1, points_3d, _ = self.PnP(points_3d, feature_1, self.img_obj.K,
                                                 np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)
        print("First Points 3D After PnP: ", points_3d)

        # 这行代码计算了总的图像数量，减去了两个的原因可能是因为在之前的代码中使用了两张图像。
        print(len(self.img_obj.image_list))
        print(self.img_obj.image_list)
        total_images = len(self.img_obj.image_list) - 2
        # 这里使用 np.hstack 函数将两个相机位姿矩阵 pose_array 和 pose_0 进行水平拼接，
        # 然后再将得到的结果与 pose_1 进行水平拼接。最终，pose_array 包含了所有图像的相机位姿信息。
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))
        print("Total Pose Array: ", pose_array)

        threshold = 0.5
        # 使用 tqdm 库中的 tqdm(range(total_images)) 来创建一个进度条，遍历从第三张图像开始到倒数第二张图像的所有图像
        for i in tqdm(range(total_images)):
            # 读取并下采样第 i + 2 张图像，将其保存在 image_2 中
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
            # 使用 self.find_features 函数提取当前图像 (image_2) 和前一张图像 (image_1) 的特征点。
            features_cur, features_2 = self.find_features(image_1, image_2)

            # 这一部分的逻辑是在处理非第一张图像时执行的。首先，使用之前的相机姿态和特征点信息进行三角化，得到三维点云坐标。
            # 然后，对 feature_1 进行转置操作，以便后续处理。接着，将三维点云坐标从齐次坐标转换为普通的三维坐标。
            # 最后，将 points_3d 设置为 points_3d[:, 0, :]。这一系列的操作可能是为了调整数据的形状以适应后续的计算或处理。
            if i != 0:
                feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]

            # 使用 self.common_points 函数获取共视的特征点及相应的特征点筛选掩码。
            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(feature_1, features_cur, features_2)
            # 根据筛选的掩码 cm_points_1 从 features_2 和 features_cur 中提取共视点对应的特征点
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            # 使用 self.PnP 函数对共视点进行PnP求解，得到旋转矩阵 rot_matrix 和平移矩阵 tran_matrix，以及更新后的三维点坐标 points_3d 和特征点坐标 cm_points_cur。
            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(points_3d[cm_points_0],
                                                                                      cm_points_2, self.img_obj.K,
                                                                                      np.zeros((5, 1),
                                                                                               dtype=np.float32),
                                                                                      cm_points_cur, initial=0)
            # 使用 PnP 求解得到的旋转矩阵和平移矩阵来更新变换矩阵 transform_matrix_1，
            # 然后使用相机内部参数 self.img_obj.K 和更新后的变换矩阵 transform_matrix_1 计算第三张图像的相机姿态矩阵 pose_2。
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            # 使用 self.reprojection_error 函数计算重投影误差，这里 homogenity=0 表示输出的三维点云坐标不使用齐次坐标。
            error, points_3d = self.reprojection_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K,
                                                       homogenity=0)

            # 这部分代码再次使用 self.triangulation 函数进行三角化，并使用 self.reprojection_error 计算重投影误差，
            # 这次 homogenity=1 表示输出的三维点云坐标使用齐次坐标。最后，将计算得到的相机姿态矩阵 pose_2 拼接到 pose_array 中。
            cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K,
                                                       homogenity=1)
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))
            # takes a long time to run
            # 是否启用捆绑调整（Bundle Adjustment），如果启用，则进行捆绑调整，否则直接使用已经计算得到的三维点坐标。
            if enable_bundle_adjustment:
                # 调用 self.bundle_adjustment 函数进行捆绑调整。该函数接受三维点坐标、特征点掩码、
                # 变换矩阵、相机内部参数和阈值作为输入，并返回经过调整的三维点坐标、新的特征点掩码以及更新后的变换矩阵。
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, cm_mask_1,
                                                                                  transform_matrix_1, self.img_obj.K,
                                                                                  threshold)
                # 使用更新后的变换矩阵 transform_matrix_1 和相机内部参数 self.img_obj.K 计算新的相机姿态矩阵 pose_2，然后计算新的重投影误差，并打印输出。
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K,
                                                           homogenity=0)
                print("Bundle Adjusted error: ", error)
                # 将计算得到的三维点坐标 points_3d 按行堆叠到 total_points 中。然后，从 cm_mask_1 中提取点坐标信息，
                # 作为索引从 image_2 中获取颜色信息，将颜色信息按行堆叠到 total_colors 中。这样就得到了所有图像中的三维点坐标和颜色信息。
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                # 如果捆绑调整未启用，直接使用 points_3d[:, 0, :] 作为三维点坐标。
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector))

                # 在迭代过程中，保存当前图像的变换矩阵 transform_matrix_1 和相机姿态 pose_1，以备后续使用
            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            # 使用 plt.scatter 绘制散点图，横坐标表示迭代次数 i，纵坐标表示当前的重投影误差 error。使用 plt.pause 产生短暂的暂停，使得图表能够更新并显示
            plt.scatter(i, error)
            plt.pause(0.05)

            # 将当前的图像 image_2 赋值给上一张图像 image_1，当前的特征 features_cur 赋值给上一张特征 feature_1。这样可以在下一轮迭代中使用。
            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)
            # 使用 OpenCV 的 cv2.imshow 函数显示当前图像，其中 self.img_obj.image_list[0].split('\\')[-2] 用于提取
            # 图像路径中的上一级文件夹名，并作为窗口的名称。cv2.waitKey(1) 用于等待键盘输入，如果按下 'q' 键则退出循环，终止程序。
            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        # 在程序结束时调用 cv2.destroyAllWindows() 来关闭所有打开的显示窗口。
        cv2.destroyAllWindows()

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.img_obj.path, total_points, total_colors)
        save_sfm_result_to_pcd(total_points, total_colors, self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2] + '_pointcloud.pcd')

        print("Completed Exiting ...")
        np.savetxt(self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2] + '_pose_array.csv',
                   pose_array, delimiter='\n')

        pcd = o3d.io.read_point_cloud('res\\myCV_pointcloud.pcd')

        # Reconstruct dense point cloud using SFM
        # (replace this with your own SFM code)
        # 这部分是一个占位符，用于模拟通过结构光流（SFM）等技术生成的稠密点云。实际上，你应该用你自己的稠密点云生成代码替代这里的随机生成。
        dense_points = total_points

        # 计算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 获取法向量
        normals = np.asarray(pcd.normals)

        # Compute neighbor information for each point using Open3D
        # 在指定半径内查找邻近点
        kdtree = KDTree(total_points)
        k = kdtree.query_ball_point(total_points[:1], r=0.3)
        neighbors = np.asarray(k)
        print(neighbors)

        # Reconstruct polygons using normal vectors and neighbor information
        polygons = []

        for i in range(total_points.shape[0]):
            neighbors = neighbors[i]
            normal = normals[i]
            plane = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0)
            rotation_matrix = Rotation.from_matrix(normal).as_matrix()
            plane.rotate(rotation_matrix, center=(0, 0, 0))
            plane.translate(pcd.points[i])
            polygons.append(o3d.geometry.Polygon(plane.sample_points_uniformly(number_of_points=100)))
        # Extend polygons to fill in gaps and remove small areas
        extended_polygons = []
        for polygon in polygons:
            extended_polygon = polygon.create_from_triangle_mesh().translate(polygon.center)
            extended_polygons.append(extended_polygon)

        # Filter out polygons with low surface area
        filtered_polygons = []
        for polygon in extended_polygons:
            if polygon.get_area() > 0.01:
                filtered_polygons.append(polygon)

        # Generate depth map from dense point cloud and polygons
        depth_map = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.01).compute_vertex_normals().sample_points_uniformly(500).point_arrays['depth']

        # Extract polygons from depth map
        polygon_points = []
        for polygon in filtered_polygons:
            polygon_points.append(np.array(depth_map.crop(polygon.get_axis_aligned_bounding_box()).points))

        # Save polygons to file
        np.save('polygon_points.npy', polygon_points)

if __name__ == '__main__':
    # sfm = Sfm("Datasets\\Herz-Jesus-P8")
    sfm = Sfm("Datasets\\myCV")
    sfm()
