import numpy as np
from dijkstar import Graph, find_path
from sklearn.metrics import pairwise_distances_argmin_min


def readData():
    # Input or create graph data
    graph = np.load("D:\\Users\\zmche\\Desktop\\pythonProject\\MNIST\\train_images_reduced.npy")
    return graph

def save_C0_as_numpy(C_0, filename="C_0_Kmeans_plus_plus.npy"):
    # 提取节点信息
    data = np.array([[node.id, len(node.points)] for node in C_0])

    # 保存为 .npy 文件
    np.save(filename, data)
    print(f"Saved C_0 to {filename}")

# 自定义距离函数（图的最短距离）
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)  # 计算欧氏距离


# 自定义 KMeans++
def kmeans_plus_plus_initialization(graph, X, n_clusters):
    n_samples = X.shape[0]
    centers = []

    # 随机选择第一个中心点
    centers.append(X[np.random.randint(n_samples)])

    for _ in range(1, n_clusters):
        # 计算每个点到最近中心点的距离
        distances = np.array([
            min(euclidean_distance(x, center) for center in centers)
            for x in X
        ])

        # 使用概率分布选择下一个中心点
        probabilities = distances ** 2 / np.sum(distances ** 2)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        next_center_index = np.searchsorted(cumulative_probabilities, r)
        centers.append(X[next_center_index])

    return np.array(centers)


# 修改的 CustomKMeans
class CustomKMeans:
    def __init__(self, graph, n_clusters=8, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.graph = graph

    def fit(self, X, y=None):
        """重写 fit 方法，使用自定义距离度量"""
        X = np.array(X)  # 确保数据为 numpy 数组

        # 初始化聚类中心（使用 KMeans++）
        self.cluster_centers_ = kmeans_plus_plus_initialization(self.graph, X, self.n_clusters)

        return self.cluster_centers_

# 测试
# read data
graph = readData()
print(graph)

# Initialize variables
U = graph  # List of nodes
data = np.array(U).reshape(-1, 1)

# 使用自定义 KMeans++
custom_kmeans = CustomKMeans(graph, n_clusters=2)
C_0 = custom_kmeans.fit(data)

# 保存为 .npy 文件
filename = "C_0_Kmeans_plus_plus.npy"
np.save(filename, C_0)
print(f"Saved C_0 to {filename}")


