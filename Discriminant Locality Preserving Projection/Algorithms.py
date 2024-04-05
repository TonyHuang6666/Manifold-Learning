# 导入库函数
import numpy as np
from os import listdir, path
from cv2 import imread, resize, INTER_AREA, IMREAD_GRAYSCALE
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from scipy.interpolate import interp1d

###############################PCA算法函数######################################
# PCA实现函数
def PCA(X, n_components):
    # 计算数据矩阵的均值
    mean = np.mean(X, axis=0)
    # 中心化数据矩阵
    X_centered = X - mean
    # 计算数据矩阵的协方差矩阵
    covariance_matrix = np.cov(X_centered, rowvar=False)
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # 对特征向量按特征值从大到小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # 选取前n_components个特征向量
    principal_components = sorted_eigenvectors[:, :n_components]
    # 返回投影矩阵和均值向量
    return principal_components, mean

###############################LPP算法函数######################################
#####################计算自适应knn graph####################
def compute_adaptive_k(Data):
    n = Data.shape[1]  # 样本点的数量
    distances = np.sqrt(np.sum((Data.T[:, :, None] - Data.T[:, :, None].T) ** 2, axis=1)) 
    sorted_distances = np.sort(distances, axis=1)  # 对距离矩阵的每一行进行排序
    adaptive_k = np.zeros((n, 1))
    # 对每行距离进行插值和求导
    for i in range(n):
        # 对距离进行插值，生成连续的函数
        f = interp1d(np.arange(n), sorted_distances[i], kind='linear')
        # 求导
        df = np.gradient(f(np.arange(n)))  # 计算函数的导数
        # 寻找导数为1的位置
        idx = np.where(df <= 1)[0][0]
        # 将索引保存为每个数据点的k值
        adaptive_k[i] = idx
    return adaptive_k, sorted_distances, distances

# 根据adaptive_k中每一行的k值和sorted_distances对每个数据点构建knn graph
def adaptive_knn_graph(Data):
    adaptive_k, sorted_distances, distances = compute_adaptive_k(Data)
    n = Data.shape[1]  
    knn_adjacency_matrix = np.zeros((n, n))  
    for i in range(n):
        indices = np.argsort(sorted_distances[i])[:int(adaptive_k[i])]
        knn_adjacency_matrix[i, indices] = 1
        knn_adjacency_matrix[indices, i] = 1
    return knn_adjacency_matrix, adaptive_k, sorted_distances, distances 
#####################计算自适应knn graph####################

#####################计算经典knn graph####################
def knn_graph(Data, method, k):
    n = Data.shape[1]  
    knn_adjacency_matrix = np.zeros((n, n))  
    distances = np.sqrt(np.sum((Data.T[:, :, None] - Data.T[:, :, None].T) ** 2, axis=1))
    if method == 'epsilon':
        return knn_adjacency_matrix, distances
    indices = np.argsort(distances, axis=1)[:, 1:k+1]
    for i in range(n):
        knn_adjacency_matrix[i, indices[i]] = 1
        knn_adjacency_matrix[indices[i], i] = 1
    return knn_adjacency_matrix, distances
#####################计算经典knn graph####################

#####################计算epsilon graph####################
def compute_avg_radius(n, distances): 
    radius = np.zeros(n)
    for i in range(n):
        avg_radius = np.mean(distances[:, i])  # 修改计算每个数据点的平均邻域半径的方式
        radius[i] = avg_radius
    return radius

def compute_knn_average_radius(sorted_distances, k):
    avg_knn_distances = np.mean(sorted_distances[:, 1:k+1], axis=1)  # 计算每个数据点的前k个距离的平均值作为半径
    return avg_knn_distances
#####################计算epsilon graph####################

def compute_neighborhood_matrix(Data, method, k):
    n = Data.shape[1]  # 获取样本点的数量
    distances = np.zeros((n, n))
    sorted_distances = np.zeros((n, n))
    adaptive_k = np.zeros((n, 1))
    if method == 'knn':
        knn_adjacency_matrix, distances = knn_graph(Data, method, k)
        return knn_adjacency_matrix, distances
    elif method == 'adaptive_knn':
        knn_adjacency_matrix, adaptive_k, sorted_distances, distances = adaptive_knn_graph(Data)
        return knn_adjacency_matrix, distances
    adjacency_matrix = np.zeros((n, n))
    radius = compute_knn_average_radius(sorted_distances, k)
    for i in range(n):
        neighbors = np.where(distances[:, i] <= radius[i])[0]  
        adjacency_matrix[i, neighbors] = 1
        adjacency_matrix[neighbors, i] = 1
    return adjacency_matrix, distances

def construct_weight_matrix(Data, method, k,t):
    n = Data.shape[1]  
    Weight_matrix = np.zeros((n, n))
    adjacency_matrix, distances = compute_neighborhood_matrix(Data, method, k)
    similarity_matrix = np.exp(-distances ** 2 / t)
    i_indices, j_indices = np.where(adjacency_matrix == 1)
    Weight_matrix[i_indices, j_indices] = similarity_matrix[i_indices, j_indices]
    Weight_matrix[j_indices, i_indices] = similarity_matrix[i_indices, j_indices]
    Weight_matrix += np.exp(-distances ** 2 / t)
    return Weight_matrix

def Best_weight_matrix(Data, k, t, weight_knn, weight_epsilon):
    n = len(Data)
    best_weight_matrix = np.zeros((n, n))
    # 计算k最近邻矩阵的权重矩阵和 epsilon 邻域矩阵
    knn_weight_matrix = construct_weight_matrix(Data, 'knn', k, t)
    epsilon_weight_matrix = construct_weight_matrix(Data, 'epsilon', k, t)
    
    # 加权平均计算
    best_weight_matrix = weight_knn * knn_weight_matrix + weight_epsilon * epsilon_weight_matrix
    return best_weight_matrix

def LPP(Data, d, method, k, t):
    if method == 'knn_epsilon':
        Weight_matrix = Best_weight_matrix(Data, k, t, 0.5, 0.5)
    else:
        Weight_matrix = construct_weight_matrix(Data, method, k, t)
    Degree_matrix = np.diag(np.sum(Weight_matrix, axis=1))
    Laplacian_matrix = Degree_matrix - Weight_matrix
    objective_value = np.dot(np.dot(Data, Laplacian_matrix), Data.T)  # 计算目标函数
    eigenvalues, eigenvectors = eigs(objective_value, k=d+1)
    sorted_indices = np.argsort(eigenvalues.real)
    selected_indices = sorted_indices[1:d + 1]
    selected_eigenvectors = eigenvectors.real[:, selected_indices]
    return selected_eigenvectors    
    
###############################LPP算法函数######################################



###############################MLDA算法函数######################################

# 计算每个类别的均值矩阵
def compute_classes_mean_matrix(train_data, train_labels):
    num_classes = len(np.unique(train_labels))  # 类别数量
    num_samples_per_class = train_data.shape[0] // num_classes  # 每个类别的样本数
    num_features = train_data.shape[1]  # 每个样本的特征维度
    means = np.zeros((num_classes, num_features))  # 存储每个类别的均值矩阵
    for i in range(1, num_classes + 1):  # 遍历每个类别
        temp_indices = np.where(train_labels == i)[0]  # 获取当前类别的训练数据索引
        temp_sum = np.sum(train_data[temp_indices], axis=0)  # 计算当前类别的特征值总和
        means[i-1] = temp_sum / num_samples_per_class  # 计算当前类别的均值
    return means  # 返回每个类别的均值矩阵

# 计算所有类别的整体均值矩阵
def compute_overall_mean_matrix(classes_means):
    overall_mean = np.mean(classes_means, axis=0)  # 计算所有类别的整体均值
    return overall_mean.reshape(-1, 1)  # 返回整体均值矩阵（转置）

# 计算中心类别矩阵
def compute_center_class_matrix(train_data, train_labels, classes_means):
    Z = np.zeros_like(train_data)  # 初始化中心类别矩阵
    
    for i in range(train_data.shape[0]):  # 遍历训练数据
        class_index = int(train_labels[i]) - 1  # 获取当前样本所属类别的索引
        Z[i] = train_data[i] - classes_means[class_index]  # 计算中心类别矩阵
        
    return Z  # 返回中心类别矩阵

# 计算类间散布矩阵
def compute_between_class_scatter_matrix(classes_means, overall_mean):
    n = 5  # 训练集与测试集的比例
    Sb = np.zeros((classes_means.shape[1], classes_means.shape[1]))  # 初始化类间散布矩阵
    for i in range(classes_means.shape[0]):  # 遍历每个类别的均值矩阵
        Sb = np.add(Sb, n * ((classes_means[i] - overall_mean) * (classes_means[i] - overall_mean).T))  # 计算类间散布矩阵
    return Sb  # 返回类间散布矩阵

# 计算类内散布矩阵
def compute_class_scatter_matrix(Z):
    Sw = np.dot(Z.T, Z)  # 计算类内散布矩阵
    return Sw  # 返回类内散布矩阵

def MLDA(train_data, train_labels, faceshape, d):
    # 计算每个类别的均值矩阵
    classes_means = compute_classes_mean_matrix(train_data, train_labels)
    # 计算所有类别的整体均值矩阵
    overall_mean = compute_overall_mean_matrix(classes_means)
    # 计算中心类别矩阵
    Z = compute_center_class_matrix(train_data, train_labels, classes_means)
    # 计算类间散布矩阵
    Sb = compute_between_class_scatter_matrix(classes_means, overall_mean)
    # 计算类内散布矩阵
    Sw = compute_class_scatter_matrix(Z)
    # 计算投影矩阵W
    W_value = np.dot(np.linalg.inv(Sw), Sb)  
    # 计算广义特征值问题的特征值和特征向量，提取前d个最大特征值对应的特征向量
    eigen_values, eigen_vectors = eigh(W_value, eigvals=((faceshape[0] * faceshape[1]-d),(faceshape[0] * faceshape[1]-1)))  # 计算特征值和特征向量
    return eigen_vectors, overall_mean, classes_means, Z, Sb, Sw, W_value

###############################MLDA算法函数######################################



###############################DLPP算法函数######################################

# 计算每个类别的权重矩阵，度矩阵和拉普拉斯矩阵
def DLPP_LPP(train_data, method, threshold, k, t):
    Data = train_data.T
    if method == 'knn_epsilon':
        Weight_matrix = Best_weight_matrix(Data, threshold, k, t, 0.5, 0.5)
    else:
        Weight_matrix = construct_weight_matrix(Data, method, threshold, k, t)
    Degree_matrix = np.diag(np.sum(Weight_matrix, axis=1))
    Laplacian_matrix = Degree_matrix - Weight_matrix
    return Laplacian_matrix, Data

# 计算每个类别的均值矩阵
def DLPP_MLDA(train_data, train_labels, d):
    classes_means = compute_classes_mean_matrix(train_data, train_labels)
    return classes_means.T

def DLPP(train_data, train_labels, d, lpp_method, threshold, k, t):
    # Step 1: 使用MLDA进行特征提取
    F = DLPP_MLDA(train_data, train_labels, d)
    # Step 2: 使用LPP进行特征提取
    L, X = DLPP_LPP(train_data, lpp_method, threshold, k, t)
    # Step 3: 计算权重矩阵B
    num_classes = len(np.unique(train_labels))  # 计算训练集中的类别数
    B = np.zeros((num_classes, num_classes))  # 初始化权重矩阵B
    # 遍历每对类别，计算其对应的权重
    for i in range(num_classes):  # 遍历每个类别
        for j in range(num_classes):  # 再次遍历每个类别
            if i != j:  # 如果类别不相同
                fi = F[i]  # 获取第i个类别的平均脸
                fj = F[j]  # 获取第j个类别的平均脸
                # 计算第i类别和第j类别平均脸之间的欧氏距离，并将其应用于高斯核函数，计算权重
                B[i, j] = np.exp(-np.linalg.norm(fi - fj) ** 2 / t)
    # Step 4: 计算E和H矩阵
    E = np.diag(np.sum(B, axis=1))
    H = E - B
    # Step 5: 计算目标函数的分母和分子
    denominator = np.dot(np.dot(F, H), F.T)
    numerator = np.dot(np.dot(X, L), X.T)
    # Step 6: 分式
    objective_value = numerator / denominator
    # Step 7: 求解广义特征值问题的特征值和特征向量
    eigenvalues, eigenvectors = eigs(objective_value, k=d+1)
    sorted_indices = np.argsort(eigenvalues.real)
    selected_indices = sorted_indices[1:d + 1]  
    selected_eigenvectors = eigenvectors.real[:, selected_indices] 
    return F, L, B, objective_value, selected_eigenvectors

###############################DLPP算法函数######################################



# 读取数据集
def read_images(dataset_dir, target_size=None):
    data = []  # 存储图像数据的列表
    labels = []  # 存储标签的列表
    faceshape = [] # 存储图像形状
    for class_dir in listdir(dataset_dir):  # 遍历数据集文件夹中的文件夹（每个文件夹代表一个类别）
        class_path = path.join(dataset_dir, class_dir)  # 类别文件夹路径
        for file_name in listdir(class_path):  # 遍历每个类别文件夹中的图像文件
            file_path = path.join(class_path, file_name)  # 图像文件路径
            img = imread(file_path, IMREAD_GRAYSCALE)  # 读取灰度图像
            # 如果指定了目标尺寸，则缩放图像
            if target_size is not None:
                img = resize(img, target_size, interpolation=INTER_AREA)
            # 读取第一张灰度图像的大小作为图片形状
            if not faceshape:
                faceshape = img.shape
            data.append(img.flatten())  # 将图像展平并添加到数据列表中
            labels.append(int(class_dir))  # 将类别标签添加到标签列表中
    return np.array(data), np.array(labels).reshape(-1, 1), faceshape  # 返回图像数据和标签

# 训练集和测试集划分
def train_test_split(data, labels, train_test_split_ratio):
    num_samples = data.shape[0]  # 总样本数
    train_samples = int(num_samples * train_test_split_ratio)  # 训练集样本数
    
    # 洗牌算法打乱数据集
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    # 划分训练集和测试集
    train_data = data[:train_samples]
    train_labels = labels[:train_samples]
    test_data = data[train_samples:]
    test_labels = labels[train_samples:]
    
    return train_data, train_labels, test_data, test_labels

# 测试DLPP, LPP, PCA要查询的图像
def test_image(i,train_labels, test_labels, query, eigenfaces, weight_matrix):
    # 计算测试图像的权重向量
    query_weight = (eigenfaces.T @ query.reshape(-1, 1))
    # 计算测试图像权重与数据集中每个人脸权重的欧氏距离
    euclidean_distances = np.linalg.norm(weight_matrix - query_weight, axis=0)
    # 找到最佳匹配的人脸
    best_match_index = np.argmin(euclidean_distances)
    #判断是否匹配正确
    flag = True
    if train_labels[best_match_index] == test_labels[i]:
        flag = True
    else:
        flag = False
    return flag

# 测试LDA要查询的图像
def test_query_class_sample(W, query_image, j, overall_mean, train_data, train_labels, test_labels):
    # 计算查询图像的线性判别函数值,即计算 d(Q) = W^T (Q - P)
    d = np.dot(W.T, (query_image - overall_mean))
    # 计算 ||d||
    discriminant_values = []
    for i in range(train_data.shape[0]):
        train_image = train_data[i]
        train_d = np.dot(W.T, (train_image - overall_mean))
        discriminant_value = np.linalg.norm(d - train_d)
        discriminant_values.append(discriminant_value)
    # 找到匹配的样本图像索引
    best_match_index = np.argmin(discriminant_values)
    
    #判断是否匹配正确
    flag = True
    if train_labels[best_match_index] == test_labels[j]:
        flag = True
    else:
        flag = False
    # 返回匹配结果
    return flag