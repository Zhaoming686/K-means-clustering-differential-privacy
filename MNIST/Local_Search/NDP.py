import numpy as np
from dijkstar import Graph, find_path
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def readData():
    # Input or create graph data
    graph = np.load("D:\\Users\\zmche\\Desktop\\pythonProject\\MNIST\\train_images_reduced.npy")
    return graph

# 自定义距离函数（图的最短距离）
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)  # 计算欧氏距离


# 继承 KMeans 类并重写方法
class CustomKMeans(KMeans):
    def __init__(self, graph, n_clusters=8, random_state=None, Center=None):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.graph = graph
        self.cluster_centers_ = Center

    def _eucledean_dist(self, X, centers):
        """计算自定义距离"""
        n_samples = X.shape[0]
        n_clusters = centers.shape[0]
        distances = np.zeros((n_samples, n_clusters))

        for i in range(n_samples):
            for j in range(n_clusters):
                distances[i, j] = euclidean_distance(X[i], centers[j])

        return distances

    def fit(self, X, y=None):
        """重写 fit 方法，使用自定义距离度量"""
        X = np.array(X)  # 确保数据为 numpy 数组
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # 初始化聚类中心
        if self.cluster_centers_ is None:
            self.cluster_centers_ = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for _ in range(300):  # 最大迭代次数
            # 计算每个点到聚类中心的距离
            distances = self._eucledean_dist(X, self.cluster_centers_)

            # 将每个点分配给最近的中心
            labels = np.argmin(distances, axis=1)

            # 更新聚类中心为簇内距离均值最近的点
            new_centers = []
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]  # 属于簇 i 的所有点
                if len(cluster_points) > 0:
                    # 计算簇内均值
                    cluster_mean = cluster_points.mean(axis=0)
                    # 找到距离均值最近的点作为新的中心
                    closest_point_idx = np.argmin(np.linalg.norm(cluster_points - cluster_mean, axis=1))
                    new_centers.append(cluster_points[closest_point_idx])
            new_centers = np.array(new_centers)

            # 判断聚类中心是否发生变化
            if np.all(new_centers == self.cluster_centers_):
                break
            self.cluster_centers_ = new_centers

        # 计算最终的标签
        self.labels_ = labels
        return self

    def compute_total_cost(self, X):
        """计算总成本（所有点到其聚类中心的距离之和）"""
        total_cost = 0
        for i, x in enumerate(X):
            # 获取点 x 到其分配的聚类中心的距离
            center = self.cluster_centers_[self.labels_[i]]
            total_cost += euclidean_distance(x, center)
        return total_cost

    def predict(self, X):
        """根据最终的聚类中心进行预测"""
        distances = self._eucledean_dist(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)


# 测试
# read data
graph = readData()
print(graph)

# Initialize variables
# U = list(graph._data.keys())  # List of nodes
# data = np.array(U).reshape(-1, 1)
data = graph

# 选择初始化方法
C_0 = None  # 随机初始化
# C_0 = np.load("D:\\Users\\zmche\\Desktop\\pythonProject\\Initial\\C_0_Kmeans_plus_plus.npy")  # kmeans++的方法
# C_0 = np.load("D:\\Users\\zmche\\Desktop\\pythonProject\\Initial\\C_0_HST.npy")  # HST方法

# 使用自定义 KMeans 聚类
custom_kmeans = CustomKMeans(graph, n_clusters=2, random_state=None, Center=C_0)  # 算是输入数据
custom_kmeans.fit(data)

# 输出聚类标签和聚类中心
print("Cluster labels:", custom_kmeans.labels_)
print("Cluster centers:", custom_kmeans.cluster_centers_)

# 计算总成本
total_cost = custom_kmeans.compute_total_cost(data)
print("Total cost:", total_cost)


# 可视化结果
import matplotlib.pyplot as plt

plt.scatter(data, np.zeros_like(data), c=custom_kmeans.labels_, cmap='viridis')
plt.scatter(custom_kmeans.cluster_centers_, np.zeros_like(custom_kmeans.cluster_centers_), color='red', marker='x')
plt.show()

























