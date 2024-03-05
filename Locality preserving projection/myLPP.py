import numpy as np  # 导入NumPy库
from scipy import linalg  # 导入SciPy的线性代数模块

from sklearn.neighbors import kneighbors_graph, NearestNeighbors  # 导入近邻图和最近邻模块
from sklearn.utils import check_array  # 导入用于检查数组的工具
from sklearn.base import BaseEstimator, TransformerMixin  # 导入基本估计器和转换器的基类


class LocalityPreservingProjection(BaseEstimator, TransformerMixin):
    # 局部保持投影类，继承了基本估计器和转换器的基类
    def __init__(self, n_components=2, n_neighbors=5,
                 weight='adjacency', weight_width=1.0,
                 neighbors_algorithm='auto'):
        # 初始化函数，设置了局部变量
        # 其中n_components为嵌入空间的维数，n_neighbors为每个点的最近邻数
        # weight表示使用的权重函数，weight_width为热核函数的宽度
        # neighbors_algorithm为最近邻搜索算法，默认为'auto'
        # TODO: 允许半径邻居，允许预计算的权重
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weight = weight
        self.weight_width = weight_width
        self.neighbors_algorithm = neighbors_algorithm

    def fit(self, X, y=None):
        # 拟合函数，计算并设置局部保持投影的投影矩阵
        X = check_array(X)  # 检查输入数据X是否为数组形式
        W = self._compute_weights(X)  # 计算权重矩阵W
        self.projection_ = self._compute_projection(X, W)  # 计算投影矩阵
        return self

    def transform(self, X):
        # 转换函数，将输入数据X映射到低维空间
        X = check_array(X)  # 检查输入数据X是否为数组形式
        return np.dot(X, self.projection_)  # 返回映射后的数据

    def _compute_projection(self, X, W):
        # 计算局部保持投影的投影矩阵
        # TODO: 检查权重矩阵的输入；处理稀疏情况
        X = check_array(X)  # 检查输入数据X是否为数组形式

        D = np.diag(W.sum(1))  # 计算度矩阵D
        L = D - W  # 计算拉普拉斯矩阵L
        evals, evecs = eigh_robust(np.dot(X.T, np.dot(L, X)),  # 求解特征值和特征向量
                                   np.dot(X.T, np.dot(D, X)),
                                   eigvals=(0, self.n_components - 1))
        return evecs  # 返回投影矩阵

    def _compute_weights(self, X):
        # 计算权重矩阵
        X = check_array(X)  # 检查输入数据X是否为数组形式
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,  # 计算最近邻
                                      algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(X)  # 将数据拟合到最近邻模型中

        if self.weight == 'adjacency':  # 如果权重是邻接矩阵
            W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                 mode='connectivity', include_self=True)  # 构建邻接矩阵
        elif self.weight == 'heat':  # 如果权重是热核函数
            W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                 mode='distance', include_self=True)  # 构建距离矩阵
            W.data = np.exp(-W.data ** 2 / self.weight_width ** 2)  # 计算热核函数的值
        else:
            raise ValueError("Unrecognized Weight")  # 抛出错误，未识别的权重类型
        #保存权重矩阵为csv文件
        # 对称化矩阵
        # TODO: 使此更有效并保持稀疏输出
        W = W.toarray()  # 将稀疏矩阵转换为稠密矩阵
        W = np.maximum(W, W.T)  # 取矩阵的最大值，使其对称
        return W  # 返回权重矩阵


def eigh_robust(a, b=None, eigvals=None, eigvals_only=False,
                overwrite_a=False, overwrite_b=False,
                turbo=True, check_finite=True):
    # 鲁棒的求解广义特征值问题的函数
    kwargs = dict(eigvals=eigvals, eigvals_only=eigvals_only,
                  turbo=turbo, check_finite=check_finite,
                  overwrite_a=overwrite_a, overwrite_b=overwrite_b)

    # 首先检查是否为简单情况：
    if b is None:
        return linalg.eigh(a, **kwargs)  # 使用SciPy中的eigh函数求解

    # 计算b的特征分解
    kwargs_b = dict(turbo=turbo, check_finite=check_finite,
                    overwrite_a=overwrite_b)  # b作为a的操作
    S, U = linalg.eigh(b, **kwargs_b)

    # 通过b的特征分解合并a和b
    S[S <= 0] = np.inf  # 将小于等于0的特征值替换为无穷大
    Sinv = 1. / np.sqrt(S)  # 计算特征值的平方根倒数
    W = Sinv[:, None] * np.dot(U.T, np.dot(a, U)) * Sinv  # 计算合并后的矩阵W
    output = linalg.eigh(W, **kwargs)  # 对合并后的矩阵W求解特征值问题

    if eigvals_only:
        return output  # 仅返回特征值
    else:
        evals, evecs = output  # 返回特征值和特征向量
        return evals, np.dot(U, Sinv[:, None] * evecs)  # 返回计算的结果
