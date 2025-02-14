import numpy as np
from dijkstar import Graph, find_path
from sklearn.metrics import pairwise_distances_argmin_min


def readData():
    # Input or create graph data
    graph = Graph()
    graph.add_edge(1, 2, 110)
    graph.add_edge(2, 1, 110)
    graph.add_edge(2, 3, 125)
    graph.add_edge(3, 2, 125)
    graph.add_edge(3, 4, 108)
    graph.add_edge(4, 3, 108)
    return graph

def save_C0_as_numpy(C_0, filename="C_0_Kmeans_plus_plus.npy"):
    # 提取节点信息
    data = np.array([[node.id, len(node.points)] for node in C_0])

    # 保存为 .npy 文件
    np.save(filename, data)
    print(f"Saved C_0 to {filename}")

# 自定义距离函数（图的最短距离）
def custom_distance(graph, x, y):
    # 转换为整数
    x_int = x.astype(int)[0]
    y_int = y.astype(int)[0]
    return find_path(graph, x_int, y_int).total_cost


# 自定义 KMeans++
def kmeans_plus_plus_initialization(graph, X, n_clusters):
    n_samples = X.shape[0]
    centers = []

    # 随机选择第一个中心点
    centers.append(X[np.random.randint(n_samples)])

    for _ in range(1, n_clusters):
        # 计算每个点到最近中心点的距离
        distances = np.array([
            min(custom_distance(graph, x, center) for center in centers)
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
U = list(graph._data.keys())  # List of nodes
data = np.array(U).reshape(-1, 1)

# 使用自定义 KMeans++
custom_kmeans = CustomKMeans(graph, n_clusters=2)
C_0 = custom_kmeans.fit(data)

# 保存为 .npy 文件
filename = "C_0_Kmeans_plus_plus.npy"
np.save(filename, C_0)
print(f"Saved C_0 to {filename}")


