# import struct
#
# # 文件路径和图片索引（示例使用第0张图片）
# file_path = 'D:\\Users\\zmche\\Desktop\\pythonProject\\dataset\\t10k-images.idx3-ubyte'
# image_index = 0
#
# # 读取MNIST图像文件
# with open(file_path, 'rb') as f:
#     # 解析文件头（魔数、图片数量、行数、列数）
#     magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
#
#     # 定位到指定图片
#     f.seek(16 + image_index * rows * cols)
#
#     # 读取像素数据（28*28=784字节）
#     img_data = struct.unpack('B' * rows * cols, f.read(rows * cols))
#
# # 输出二维像素数据
# print("二维像素数据（28x28）：")
# for i in range(rows):
#     print(img_data[i * cols: (i + 1) * cols])
#
# # 转换并输出一维数据
# print("\n一维像素数据（784维）：")
# print(list(img_data))

import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection

# 定义函数来读取 idx 文件
def load_idx_images(file_name):
    with open(file_name, 'rb') as f:
        # 读取魔数和数据类型等信息
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # 读取所有图片数据
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# 加载训练数据集的图像
train_images = load_idx_images('D:\\Users\\zmche\\Desktop\\pythonProject\\dataset\\t10k-images.idx3-ubyte')

# 将所有图片转换为一维数据
train_images_1d = train_images.reshape(train_images.shape[0], -1)

# 创建一个GaussianRandomProjection实例，将784维降到100维
rp = GaussianRandomProjection(n_components=100)

# 使用Random Projection进行降维
train_images_reduced = rp.fit_transform(train_images_1d)

# 输出降维后的数据形状
print(train_images_reduced)
print("降维后的数据形状：", train_images_reduced.shape)

# 可以将降维后的数据可视化（例如，用前两个主成分绘制散点图）
plt.scatter(train_images_reduced[:, 0], train_images_reduced[:, 1], s=1)
plt.title('Random Projection of Images')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# 将结果保存为 .npy 文件
np.save('train_images_reduced', train_images_reduced)

# # 如果你想保存为 CSV 格式
# import pandas as pd
# df = pd.DataFrame(train_images_1d)
# df.to_csv('mnist_train_images_1d.csv', index=False, header=False)
#
# print("数据已保存为 .npy 和 .csv 格式。")







