import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt  # 导入Matplotlib库
import os

# 计算成对距离的函数
def pairwise_distances(X):
    dist_matrix = np.sqrt(np.sum((X[:, None] - X) ** 2, axis=2))
    return dist_matrix

# 计算邻接矩阵的函数（kNN）
def nearest_neighbors(X, n_neighbors):
    dist_matrix = pairwise_distances(X)
    adjacency_matrix = dist_matrix.argsort(axis=1)[:, 1:n_neighbors+1]
    return adjacency_matrix

# 构建epsilon邻域的邻接图
def epsilon_neighborhood(X, epsilon):
    n_samples = X.shape[0]
    # 初始化邻接矩阵
    adjacency_matrix = np.zeros((n_samples, n_samples), dtype=int)
    # 遍历所有节点对
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # 计算节点 i 和节点 j 之间的欧几里得距离
            distance = np.linalg.norm(X[i] - X[j])
            # 如果距离小于阈值epsilon，则连接节点 i 和节点 j
            if distance <= epsilon:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
            elif distance > epsilon:
                adjacency_matrix[i, j] = 0
                adjacency_matrix[j, i] = 0
    return adjacency_matrix

def construct_weight_matrix(X, n_neighbors, method, epsilon, t):
    n = len(X)  # 计算数据集样本点的数量
    Weight_matrix = np.zeros((n, n))  # 创建一个全零的权重矩阵，尺寸为 (n, n)
    if method == 'epsilon':  # 如果方法为epsilon邻域
        adjacency_matrix = epsilon_neighborhood(X, epsilon)
        for i in range(n):  # 对于数据集中的每个样本点 i
            for j in range(n):  # 对于数据集中的每个样本点 j
                if adjacency_matrix[i, j] == 1:
                    Weight_matrix[i, j] = np.exp(- np.linalg.norm(X[i] - X[j]) ** 2 / t)  # 使用热核方法计算权重
                    Weight_matrix[j, i] = np.exp(- np.linalg.norm(X[j] - X[i]) ** 2 / t)  #权重矩阵为对称矩阵
                elif adjacency_matrix[i, j] == 0:
                    Weight_matrix[i, j] = 0
                    Weight_matrix[j, i] = 0
    elif method == 'knn':  # 如果方法为k最近邻
        dist_matrix = pairwise_distances(X)  # 计算数据集中样本点之间的距离
        knn_matrix = nearest_neighbors(X, n_neighbors)  # 计算每个样本点的 n_neighbors 个最近邻索引
        for i in range(len(X)):  # 对于数据集中的每个样本点 i
            for j in knn_matrix[i]:  # 对于样本点 i 的 n_neighbors 个最近邻点 j
                Weight_matrix[i][int(j)] = np.exp(- dist_matrix[i][int(j)] ** 2 / t)  # 使用热核方法计算权重
                Weight_matrix[int(j)][i] = np.exp(- dist_matrix[int(j)][i] ** 2 / t)  #邻接矩阵为对称矩阵
    return Weight_matrix  # 返回构建完成的邻接矩阵

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

# LPP算法
def LPP(X, n_neighbors, n_components, method, epsilon, t ):
    # Step 1: 构建权重矩阵
    Weight_matrix = construct_weight_matrix(X, n_neighbors, method, epsilon, t )
    # Step 2: 计算度矩阵和拉普拉斯矩阵
    Degree_matrix = np.diag(np.sum(Weight_matrix, axis=1))
    Laplacian_matrix = Degree_matrix - Weight_matrix
    # Step 4: 进行特征映射
    selected_eigenvectors = eigen_mapping(Laplacian_matrix, n_components)
    return selected_eigenvectors

# 示例数据
Data = np.load('D:/OneDrive - email.szu.edu.cn/Manifold Learning/swiss roll-1000.npy')

# 将 Data 转置
Data_T = np.transpose(Data)

# 设置降维参数
n_neighbors = 150
n_components = 2
method = 'epsilon' # 方法选择knn或者epsilon


# 调用LPP函数的同时让epsilon从0.1到10变化，每次变化0.1，t从10到100变化，每次变化10，保存出不同参数下的投影图
# 设置 epsilon 和 t 的范围和步长
epsilon_range = [i for i in range(1, 10)]  # 从1到10，每次变化1
t_range = list(range(10, 16))  # 从10到15，每次变化1

# 循环遍历不同的 epsilon 和 t
for epsilon in epsilon_range:
    for t in t_range:
        # 执行 LPP 算法
        lpp_embeddings = LPP(Data_T, n_neighbors=n_neighbors, n_components=n_components, method=method, epsilon=epsilon, t=t)
        # 绘制投影图
        plt.figure(figsize=(14,10))
        plt.scatter(lpp_embeddings[:, 0], lpp_embeddings[:, 1], c=plt.cm.jet((Data_T[:,0]**2+Data_T[:,2]**2)/100), s=200, lw=0, alpha=1)
        plt.title('LPP with epsilon = ' + str(epsilon) + ' and t = ' + str(t), size=25)
        file_name = 'LPP_epsilon_' + str(epsilon) + '_t_' + str(t) + '.png'
        file_path = './output/' + file_name
        plt.savefig(file_path)



