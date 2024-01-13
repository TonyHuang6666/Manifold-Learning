import numpy as np #导入numpy库
import matplotlib.pyplot as plt #导入matplotlib库
from scipy.linalg import solve  # 导入 solve 函数
from scipy.sparse import csr_matrix  # 导入函数

def pairwise_distances(X):
    dist_matrix = np.sqrt(np.sum((X[:, None] - X) ** 2, axis=2))
    return dist_matrix

# 计算最近邻的索引
def nearest_neighbors(X, k):
    dist_matrix = pairwise_distances(X)
    return dist_matrix.argsort(axis=1)[:, 1:k+1]

# 计算权重矩阵
def barycenter_weights(X, Z, reg=1e-3):
    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]
        G = np.dot(C, C.T)
        trace = np.trace(G)
        R = reg * trace if trace > 0 else reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v)
        B[i, :] = w / np.sum(w)
    return B

# 计算邻居的索引（避免重复计算）
def get_neighbors_indices(X, n_neighbors):
    knn_matrix = nearest_neighbors(X, n_neighbors)
    n_samples = X.shape[0]
    knn_indices = np.zeros((n_samples, n_neighbors), dtype=int)
    def find_indices(i):
        knn_indices[i] = knn_matrix[i]
    for i in range(n_samples):
        find_indices(i)
    return knn_indices

# 计算权重矩阵的函数
def barycenter_kneighbors(X, n_neighbors, reg=1e-3):
    n_samples = X.shape[0]
    ind = get_neighbors_indices(X, n_neighbors)
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples))

# 低维嵌入算法实现
def Embed(Weight_Matrix, n_components):
    # 计算矩阵 M
    I = np.identity(Weight_Matrix.shape[1])
    M = (I - Weight_Matrix).T @ (I - Weight_Matrix)
    # 计算矩阵M的特征值及特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # 将特征向量按照特征值从小到大排序
    sorted_indices = np.argsort(eigenvalues)
    # 选择最小的 n_components 个特征向量
    selected_indices = sorted_indices[:n_components + 1]
    # 丢弃最小的特征向量
    selected_indices = selected_indices[1:]
    # 返回特征向量
    selected_eigenvectors = eigenvectors[:, selected_indices]
    return selected_eigenvectors

def LLE(data, n_neighbors, n_components):
    # 计算权重矩阵
    Weight_Matrix = barycenter_kneighbors(data, n_neighbors)  # 计算权重矩阵
    # 保存权重矩阵
    # 计算降维后的结果
    embeddings = Embed(Weight_Matrix, n_components)  # 计算降维后的嵌入结果
    return embeddings  # 返回嵌入结果

Data=np.load('./swiss roll-1000.npy')
Data_T = np.transpose(Data)

# 绘制三维数据集
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(Data[0, :], Data[1, :], Data[2, :], c=plt.cm.jet((Data[0, :]**2 + Data[2, :]**2) / 100), s=200, lw=0, alpha=1)
ax.set_xlim(np.min(Data[0, :]), np.max(Data[0, :]))
ax.set_ylim(np.min(Data[1, :]), np.max(Data[1, :]))
ax.set_zlim(np.min(Data[2, :]), np.max(Data[2, :]))
plt.title('swiss roll - '+ str(Data.shape[1]) + ' points', size=30)
ax.axis("off")

n_neighbors=70

# 使用 LLE 算法降维
LLE_X = LLE(Data_T, n_neighbors=n_neighbors, n_components=2)  # 使用 LLE 算法降维，将数据维度从 3D 减少到 2D

# 绘制 LLE 投影图
plt.figure(figsize=(14,10))
plt.scatter(LLE_X[:, 0].tolist(), LLE_X[:, 1].tolist(), c=plt.cm.jet((Data_T[:,0]**2+Data_T[:,2]**2)/100), s=200, lw=0, alpha=1)
plt.title('swiss roll-LLE neighbors = ' + str(n_neighbors), size=30)
plt.axis("off")

plt.show()



