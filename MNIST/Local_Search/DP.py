import random
import math
import numpy as np
from dijkstar import Graph, find_path
from sklearn.cluster import KMeans
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

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)  # 计算欧氏距离


def cost(graph, data, centers):
    centers = np.array(centers).reshape(-1, 1)
    # data根据距离归类
    data = np.array(data).reshape(-1, 1)
    # 计算每个点到聚类中心的距离
    """计算自定义距离"""
    n_samples = data.shape[0]
    n_clusters = centers.shape[0]
    distances = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            distances[i, j] = euclidean_distance(data[i], centers[j])

    # 将每个点分配给最近的中心
    labels = np.argmin(distances, axis=1)

    """计算总成本（所有点到其聚类中心的距离之和）"""
    total_cost = 0
    for i, x in enumerate(data):
        # 获取点 x 到其分配的聚类中心的距离
        center = centers[labels[i]]
        total_cost += euclidean_distance(x, center)
    return total_cost

def DP_HST_LocalSearch(graph, U, F, epsilon, T):
    # Computer the adjusted epsilon
    delta = 4 * (T + 1)
    epsilon_prime = epsilon / delta

    centers = []
    costs = []
    for i in range(1, T + 1):
        # Select (x, y) from F and V \ F with probability proportional to exp(-epsilon' * cost)
        V_minus_F = [v for v in U if v not in F]
        x = random.choice(F)
        y = random.choice(V_minus_F)

        # # Compute the new cost after swapping x with y
        # new_F = F.copy()
        # new_F.remove(x)
        # new_F.append(y)
        # swap_cost = cost(graph, U, new_F)

        # 计算交换 x 和 y 后的代价
        new_F = F.tolist()  # 转换为列表
        new_F.remove(x)
        new_F.append(y)
        new_F = np.array(new_F)  # 转换回 NumPy 数组
        swap_cost = cost(graph, U, new_F)

        # Determine whether to accept the new center set based on exponential mechanism
        acceptance_prob = math.exp(-epsilon_prime * swap_cost)
        if random.random() < acceptance_prob:
            F = new_F

        centers.append(F)
        costs.append(cost(graph, U, F))

    # Final selection of Fj based on probability
    # probabilities = [math.exp(-epsilon_prime * c) for c in costs]
    # total_prob = sum(probabilities)
    # probabilities = [p / total_prob for p in probabilities]
    probabilities = np.exp(-epsilon_prime * np.array(costs))
    probabilities /= probabilities.sum()  # 归一化

    # Choose the final set Fj based on probabilities
    selected_index = random.choices(range(len(costs)), weights=probabilities, k=1)[0]
    print(centers[selected_index])
    print(costs[selected_index])

    return centers[selected_index]

# 测试
graph = readData()
print(graph)

# Initialize variables
# U = list(graph._data.keys())  # List of nodes
# data = np.array(U).reshape(-1, 1)
data = graph

# 选择初始化方法
k = 2
C_0 = data[np.random.choice(len(data), k, replace=False)]  # 随机选 k 个, 随机初始化
# C_0 = np.load("D:\\Users\\zmche\\Desktop\\pythonProject\\Initial\\C_0_Kmeans_plus_plus.npy")  # kmeans++的方法
# C_0 = np.load("D:\\Users\\zmche\\Desktop\\pythonProject\\Initial\\C_0_DP_HST.npy")  # DP_HST方法

# Initialize the centers
U = data
F = C_0
epsilon = 1
T = 300
centers = DP_HST_LocalSearch(graph, U, F, epsilon, T)

# 输出聚类标签和聚类中心
print("Cluster centers:", centers)

# 计算总成本
total_cost = cost(graph, data, centers)
print("Total cost:", total_cost)











