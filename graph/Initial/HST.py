from random import choice
from dijkstar import Graph, find_path, NoPathError
import copy
import math
import random
import numpy as np

idNode = {}
global_idNode = 0

class HST:
    def __init__(self):
        self.L = 0
        self.nodes = []
        self.L_nodes = {}

    def setL(self, L):
        self.L = L

    def setL_nodes(self, node):
        if self.L not in self.L_nodes:  # Check if the key exists
            self.L_nodes[self.L] = []  # Initialize as an empty list if not
        self.L_nodes[self.L].append(node)  # Add node to the list

    def display(self):
        # Display the HST
        print("\nHST Construction with Children:")
        for level in self.L_nodes:
            print(f"Level {level}:")
            for node in self.L_nodes[level]:
                node.display()


class node:
    def __init__(self, id, points):
        self.id = id
        self.points = points
        self.children = []

    def display(self, level=0):
        # Display node ID, points, and recursively display children
        indent = "  " * level  # Indentation based on the depth of the node
        print(f"{indent}Node ID: {self.id}, Points: {self.points}")
        for child in self.children:
            idNode.get(child).display(level + 1)  # Recurse for child nodes

    def setChildren(self, child):
        self.children.append(child.id)

    def getChildren(self):
        return self.children

    def getPoints(self):
        return self.points

    def copy(self):
        # Create a deep copy of the current node, including its children
        copied_node = node(self.id, copy.deepcopy(self.points))
        if self.children is not None:
            copied_node.children = [child.copy() for child in self.children]
        return copied_node


# 构造HST
def build_two_HST(T, graph, delta, currentNode, L):
    global global_idNode
    global idNode

    U = currentNode.getPoints().copy()

    r = delta / (2 ** (2 - L))
    T.setL(L)  # Set the current level

    clusters = []
    while U:  # Process nodes until U is empty
        v = U.pop(0)  # Take the first element
        C_v = [v]
        indexToRemove = []
        for index, u in enumerate(U):
            if u == v:  # 不能是当前节点
                indexToRemove.append(index)
                continue

            d_uv = 10000000

            try:
                d_uv = find_path(graph, u, v).total_cost  # 总共的消耗
            except NoPathError as e:
                # 如果路径不存在，捕获异常并处理
                print(f"No path could be found from '{u}' to '{v}': {e}")

            # d_uv = find_path(graph, u, v).total_cost  # 总共的消耗
            if d_uv <= r:
                C_v.append(u)
                indexToRemove.append(index)  # 记录需要删除的索引
                print(f"Index to remove: {index}, Value: {u}")
        # 按降序删除索引
        for idx in sorted(indexToRemove, reverse=True):
            del U[idx]  # 删除对应索引的元素
        print(f"After removal, U: {U}")

        # Add C_v as a new node
        global_idNode += 1
        newNode = node(global_idNode, C_v.copy())
        idNode[global_idNode] = newNode
        T.setL_nodes(newNode)

        # Add this cluster to the list
        clusters.append(newNode)

        # add this cluster to the HST's parent
        currentNode.setChildren(newNode)

    # Recursive call for each cluster
    for cluster in clusters:
        if len(cluster.getPoints()) > 1 and L > 0:  # Only recurse if the cluster has more than one node
            build_two_HST(T, graph, delta, cluster, L - 1)

def readData():
    # Input or create graph data
    graph = Graph()
    # file = open("D:\\Users\\zmche\\Desktop\\pythonProject\\dataset\\citeseer.txt").readlines()
    # for line in file:
    #     if line[0] == 'e':
    #         _, vertex1, vertex2, weight = line.split(" ")
    #         vertex1 = int(vertex1)
    #         vertex2 = int(vertex2)
    #         weight = float(weight)
    #         graph.add_edge(vertex1, vertex2, weight)
    #         graph.add_edge(vertex2, vertex1, weight)  # 相反方向再来一次
    # return graph
    graph.add_edge(1, 2, 110)
    graph.add_edge(2, 1, 110)
    graph.add_edge(2, 3, 125)
    graph.add_edge(3, 2, 125)
    graph.add_edge(3, 4, 108)
    graph.add_edge(4, 3, 108)
    return graph

def save_C0_as_numpy(C_0, filename="C_0_HST.npy"):
    # 提取节点信息
    data = np.array([node.points for node in C_0])

    # 保存为 .npy 文件
    np.save(filename, data)
    print(f"Saved C_0 to {filename}")

def getDelta(graph, U):
    delta = -100

    n = len(U)

    # 抽取的点数：根号n
    num_samples = math.floor(math.sqrt(n))  # 或者 math.ceil(math.sqrt(n))，取整方式可按需求调整

    # 随机抽取点
    sampled_points = random.sample(U, num_samples)

    for u in sampled_points:
        for v in sampled_points:
            try:
                d_uv = find_path(graph, u, v).total_cost  # the length of the road
            except NoPathError as e:
                # 如果路径不存在，捕获异常并处理
                print(f"No path could be found from '{u}' to '{v}': {e}")
            if d_uv > delta:
                delta = d_uv
    return delta

def initialHST(T):
    # HST initial algorithm
    scores = {}
    for level in T.L_nodes:
        for node in T.L_nodes[level]:
            N_v = len(node.points)
            scores[node] = N_v * (2 ** (2 - level))

    k = 2
    C_1 = []
    while len(C_1) < k:
        sorted_items_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for node, score in sorted_items_desc:
            if len(C_1) < 2:
                C_1.append(node)
                continue
            # 满了，判断里面的元素是否全部满足条件
            for v in C_1:
                # v的所有后代
                descendants = []
                children_temp = v.children
                while children_temp:
                    descendants.extend(children_temp)  # Add current level of children
                    new_children_temp = []  # 用于存储结果的临时列表
                    for n in children_temp:  # 遍历当前的 children_temp 列表
                        for child in idNode.get(n).children:  # 遍历每个节点的子节点
                            new_children_temp.append(child)  # 将子节点添加到结果列表中
                    children_temp = new_children_temp  # 更新 children_temp
                # 判断其他节点是否为其后代
                remove = False
                for vC_1 in C_1:
                    if vC_1.id == v.id:
                        continue
                    if vC_1.id in descendants:
                        remove = True
                # 删除
                if remove:
                    C_1.remove(v)

    # Find leaf
    C_0 = []
    for node in C_1:
        if len(node.children) == 0:  # 说明是一个叶子节点
            C_0.append(node)
        else:  # 找其真正的叶子节点
            descendants = []
            children_temp = node.children
            while children_temp:
                descendants.extend(children_temp)  # Add current level of children
                new_children_temp = []  # 用于存储结果的临时列表
                for n in children_temp:  # 遍历当前的 children_temp 列表
                    for child in idNode.get(n).children:  # 遍历每个节点的子节点
                        new_children_temp.append(child)  # 将子节点添加到结果列表中
                children_temp = new_children_temp  # 更新 ch
            # 找其最大的叶节点
            maxNw = -1
            maxId = -1
            for id in descendants:
                if len(idNode.get(id).children) == 0 and len(idNode.get(id).points) > maxNw:  # 是叶节点且数据点比较多
                    maxId = id
                    maxNw = len(idNode.get(id).points)
            C_0.append(idNode.get(maxId))

    # 输出C_0
    for c in C_0:
        c.display()

    return C_0


# 测试
# read data
graph = readData()
print(graph)

# Initialize variables
U = list(graph._data.keys())  # List of nodes
delta = getDelta(graph, U)
L = round(math.log2(delta))  # Initial level，若以10为底则 math.log10(100)
randomPoint = choice(U)  # Select a random point

# HST 初始化
# Construct HST root node
T = HST()
T.setL(L)  # Set initial level
nodeRoot = node(0, U)  # Root node with ID 0
idNode[0] = nodeRoot
T.setL_nodes(nodeRoot)  # Add root node to level 0

# Global ID for nodes
global_idNode = 0

# Define delta and invoke build_two_HST
build_two_HST(T, graph, delta, T.L_nodes.get(L)[0], L - 1)
T.display()
# 得到HST找出的初始点
C_0 = initialHST(T)

# 调用函数保存数据
save_C0_as_numpy(C_0)




















