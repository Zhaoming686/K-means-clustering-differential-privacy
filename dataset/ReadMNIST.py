import os
import struct
import numpy as np
import matplotlib.pyplot as plt

# 读取标签数据集
with open('D:\\Users\\zmche\\Desktop\\pythonProject\\dataset\\train-labels.idx1-ubyte', 'rb') as lbpath:
    labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)

# 读取图片数据集
with open('D:\\Users\\zmche\\Desktop\\pythonProject\\dataset\\train-images.idx3-ubyte', 'rb') as imgpath:
    images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols)

# 打印数据信息
print('labels_magic is {} \n'.format(labels_magic),
      'labels_num is {} \n'.format(labels_num),
      'labels is {} \n'.format(labels))

print('images_magic is {} \n'.format(images_magic),
      'images_num is {} \n'.format(images_num),
      'rows is {} \n'.format(rows),
      'cols is {} \n'.format(cols),
      'images is {} \n'.format(images))

# 测试取出一张图片和对应标签
choose_num = 1  # 指定一个编号，你可以修改这里
label = labels[choose_num]
image = images[choose_num].reshape(28, 28)

plt.imshow(image)
plt.title('the label is : {}'.format(label))
plt.show()