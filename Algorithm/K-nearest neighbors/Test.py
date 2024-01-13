import numpy as np

# 示例数据集 X
X = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])

# 初始化数据点之间的距离矩阵
dist_matrix = np.zeros((X.shape[0], X.shape[0]))

# 计算欧氏距离函数
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 计算每个数据点与其他点之间的距离
for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        if i != j:  # 排除相同的数据点
            # 计算数据点 i 和 j 之间的欧氏距离并存储到距离矩阵中
            dist_matrix[i, j] = euclidean_distance(X[i], X[j])

# 对距离矩阵进行排序并获取最近邻索引
k = 2  # 假设 k=2
indices = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
distances = np.sort(dist_matrix, axis=1)[:, 1:k+1]

# 输出 KNN 矩阵的最近邻索引和对应的距离
print("KNN 矩阵的最近邻索引：")
print(indices)
print("\n对应的距离：")
print(distances)
