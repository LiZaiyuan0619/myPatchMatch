from PIL import Image
import numpy as np

def convert_png_to_csv(input_path, output_path):
    # 打开PNG图像
    img = Image.open(input_path)

    # 将图像转换为numpy数组
    img_array = np.array(img)

    # 获取图像的高度和宽度
    h, w, _ = img_array.shape

    # 将深度图的RGB通道平均值作为深度值
    depth_data = np.mean(img_array, axis=2)

    # 创建一个CSV文件并写入深度数据
    with open(output_path, 'w') as csv_file:
        csv_file.write('row,column,depth\n')  # 写入CSV文件的头部

        # 遍历图像像素并将深度数据写入CSV文件
        for i in range(h):
            for j in range(w):
                depth = depth_data[i, j]
                csv_file.write(f'{i},{j},{depth}\n')

    print(f'Conversion completed. CSV file saved at: {output_path}')
# 替换以下路径为您的输入和输出路径
input_image_path = 'depth_map_color.png'
output_csv_path = 'depth_map_csv.csv'

convert_png_to_csv(input_image_path, output_csv_path)
