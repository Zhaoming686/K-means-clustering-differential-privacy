import numpy as np
import matplotlib.pyplot as plt

# 定义 x 和 y 数据点
x = np.array([2, 5, 10, 15, 20])
y_values = np.array([
    [89, 68, 52, 45, 41],
    [95, 71, 60, 50, 45],
    [98, 76, 65, 53, 49],
    [96, 75, 60, 52, 45],
    [90, 75, 60, 51, 47],
    [93, 72, 62, 54, 48]
])

# 曲线名称
labels = ["NDP-HST", "NDP-kmedian++", "NDP-rand", "DP-HST", "DP-kmedian++", "DP-rand"]
# 不同的标记样式
markers = ['o', 's', '^', 'D', 'p', '*']

# 生成 6 条单调递减曲线
plt.figure(figsize=(8, 6))
for i in range(6):
    plt.plot(x, y_values[i], marker=markers[i], label=labels[i])

# 添加标题和轴标签
plt.title('Balanced D')
plt.xlabel('k')
plt.ylabel('Initial Cost')
plt.legend()
plt.grid(True)
plt.show()
