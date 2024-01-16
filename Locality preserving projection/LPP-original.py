import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# 计算成对距离的函数
def pairwise_distances(X):
    dist_matrix = np.sqrt(np.sum((X[:, None] - X) ** 2, axis=2))
    return dist_matrix

# 计算邻接矩阵的函数（kNN）
def nearest_neighbors(X, n_neighbors):
    dist_matrix = pairwise_distances(X)
    adjacency_matrix = dist_matrix.argsort(axis=1)[:, 1:n_neighbors+1]
    return adjacency_matrix

# 计算每个数据点的平均邻域半径
def compute_average_radius(Data):
    n = Data.shape[0]
    average_radius = np.zeros(n)
    for i in range(n):
        distances = np.sqrt(np.sum((Data[i] - Data) ** 2, axis=1))
        avg_radius = np.mean(distances)
        average_radius[i] = avg_radius
    return average_radius

# 构建epsilon邻域的邻接矩阵
def epsilon_neighborhood(Data, epsilon):
    n_samples = Data.shape[0]
    adjacency_matrix = np.zeros((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        distances = np.sqrt(np.sum((Data[i] - Data) ** 2, axis=1))
        neighbors = np.where(distances <= epsilon)[0]
        adjacency_matrix[i, neighbors] = 1
        adjacency_matrix[neighbors, i] = 1
    return adjacency_matrix

# 构建基于热核方法的权重矩阵
def construct_weight_matrix(Data, method, n_neighbors, t):
    n = len(Data)
    Weight_matrix = np.zeros((n, n))
    if method == 'knn':
        dist_matrix = pairwise_distances(Data)  # 计算数据集中样本点之间的距离
        knn_matrix = nearest_neighbors(Data, n_neighbors)  # 计算每个样本点的 n_neighbors 个最近邻索引
        for i in range(len(Data)):  # 对于数据集中的每个样本点 i
            for j in knn_matrix[i]:  # 对于样本点 i 的 n_neighbors 个最近邻点 j
                Weight_matrix[i][int(j)] = np.exp(- dist_matrix[i][int(j)] ** 2 / t)  #使用热核方法计算权重
                Weight_matrix[int(j)][i] = np.exp(- dist_matrix[int(j)][i] ** 2 / t)  #邻接矩阵为对称矩阵
    elif method == 'epsilon':
        average_radius = compute_average_radius(Data)
        for i in range(n):
            adjacency_matrix = epsilon_neighborhood(Data, average_radius[i])
            for j in range(n):
                if adjacency_matrix[i, j] == 1:
                    distance = np.linalg.norm(Data[i] - Data[j])
                    Weight_matrix[i, j] = np.exp(-distance ** 2 / t)  #使用热核方法计算权重
                    Weight_matrix[j, i] = np.exp(-distance ** 2 / t)  #邻接矩阵为对称矩阵
    return Weight_matrix

# 进行特征映射
def eigen_mapping(L, n_components):
    # 计算广义特征值问题的特征向量和特征值
    eigenvalues, eigenvectors = eigh(L)
    # 将特征向量按照特征值从小到大排序
    sorted_indices = np.argsort(eigenvalues)
    # 选择最小的 n_components 个特征向量
    selected_indices = sorted_indices[:n_components + 1]
    # 丢弃最小的特征向量
    selected_indices = selected_indices[1:]
    # 返回特征向量
    selected_eigenvectors = eigenvectors[:, selected_indices]
    return selected_eigenvectors

# LPP 算法
def LPP(Data, n_components, method, n_neighbors, t):
    # Step 1: 构建基于权重矩阵
    Weight_matrix = construct_weight_matrix(Data, method, n_neighbors, t)
    # Step 2: 计算度矩阵和拉普拉斯矩阵
    Degree_matrix = np.diag(np.sum(Weight_matrix, axis=1))
    Laplacian_matrix = Degree_matrix - Weight_matrix
    # Step 3: 进行特征映射
    selected_eigenvectors = eigen_mapping(Laplacian_matrix, n_components)
    return selected_eigenvectors

# 示例数据
Data = np.load('D:/OneDrive - email.szu.edu.cn/Manifold Learning/swiss roll-1000.npy')

# 绘制三维数据集
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(Data[0, :], Data[1, :], Data[2, :], c=plt.cm.jet((Data[0, :]**2 + Data[2, :]**2) / 100), s=200, lw=0, alpha=1)
ax.set_xlim(np.min(Data[0, :]), np.max(Data[0, :]))
ax.set_ylim(np.min(Data[1, :]), np.max(Data[1, :]))
ax.set_zlim(np.min(Data[2, :]), np.max(Data[2, :]))
plt.title('Data - '+ str(Data.shape[1]) + ' points', size=30)
ax.axis("off")

# 将 Data 转置
Data_T = np.transpose(Data)

# 绘制及保存投影图
for method in ['epsilon', 'knn']:
    plt.figure(figsize=(14, 10))
    if method == 'epsilon':
        t = 10000
        lpp_embeddings = LPP(Data_T, n_components=2, method=method, n_neighbors=150, t=t)
        plt.title('LPP with t = ' + str(t), size=25)
        plt.scatter(lpp_embeddings[:, 0], lpp_embeddings[:, 1], c=plt.cm.jet((Data_T[:, 0] ** 2 + Data_T[:, 2] ** 2) / 100),s=200, lw=0, alpha=1)
    elif method == 'knn':
        n_neighbors = 150
        t = 60
        lpp_embeddings = LPP(Data_T, n_components=2, method=method, n_neighbors=n_neighbors, t=t)
        plt.title('LPP with k-Nearest Neighbors = ' + str(n_neighbors) + ' and t = ' + str(t), size=25)
        plt.scatter(lpp_embeddings[:, 0], lpp_embeddings[:, 1], c=plt.cm.jet((Data_T[:, 0] ** 2 + Data_T[:, 2] ** 2) / 100),s=200, lw=0, alpha=1)
plt.show()


