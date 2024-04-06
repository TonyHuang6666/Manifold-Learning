import time
from sys import argv, exit
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLineEdit, QComboBox, QTextEdit, QMessageBox, QDesktopWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Algorithms import *

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DLPP LPP LDA PCA 图像特征提取与识别程序")
        self.setGeometry(0, 0, 1200, 1300)
        self.center_on_screen()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # 数据集的选择与划分
        self.dataset_label = QLabel("请选择数据集文件夹:")
        self.main_layout.addWidget(self.dataset_label)
        self.default_dataset_path = "C:\\Users\Tony\\OneDrive - email.szu.edu.cn\\Manifold Learning\\Discriminant Locality Preserving Projection\\ORL"
        self.dataset_path_label = QLabel(f"数据集默认路径: {self.default_dataset_path}")  # 显示默认数据集文件夹的路径
        self.main_layout.addWidget(self.dataset_path_label)
        self.dataset_button = QPushButton("选择其他数据集")
        self.dataset_button.clicked.connect(self.select_dataset)
        self.main_layout.addWidget(self.dataset_button)

        self.train_test_split_label = QLabel("请选择训练集划分比例:")
        self.main_layout.addWidget(self.train_test_split_label)
        self.train_test_split_combo = QComboBox()
        for ratio in range(5, 100, 5):
            ratio_decimal = ratio / 100.0
            self.train_test_split_combo.addItem("{:.2f}".format(ratio_decimal))
        self.train_test_split_combo.setCurrentText("0.70")
        self.main_layout.addWidget(self.train_test_split_combo)

        #原版MNIST数据集的裁切
        self.mnist_split_label = QLabel("请选择原版MNIST数据集的裁切比例:")
        self.main_layout.addWidget(self.mnist_split_label)
        self.mnist_split_combo = QComboBox()
        for ratio in range(1, 100, 1):
            ratio_decimal = ratio / 100.0
            self.mnist_split_combo.addItem("{:.2f}".format(ratio_decimal))
        self.mnist_split_combo.setCurrentText("0.01")
        self.main_layout.addWidget(self.mnist_split_combo)

        # 选择图像缩放百分比
        self.target_size_label = QLabel("请选择图像缩放百分比:")
        self.main_layout.addWidget(self.target_size_label)
        self.target_size_combo = QComboBox()
        for percentage in range(5, 105, 5):
            self.target_size_combo.addItem(f"{percentage}%")
        self.target_size_combo.setCurrentText("35%")  # 设置初始值为x%,即长宽均为原来的x%且取整
        self.main_layout.addWidget(self.target_size_combo)

        # 输入降维后的维度
        self.d_label = QLabel("请输入降维后的维度d:")
        self.main_layout.addWidget(self.d_label)
        self.d_input = QLineEdit()
        self.d_input.setText("70")  # 默认值为70
        self.main_layout.addWidget(self.d_input)

        # 选择降维方法
        self.method_label = QLabel("请选择数据降维方法:")
        self.main_layout.addWidget(self.method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItem("DLPP")
        self.method_combo.addItem("LPP")
        self.method_combo.addItem("MLDA")
        self.method_combo.addItem("PCA")
        self.method_combo.setCurrentText("DLPP")  # 设置初始值为当前选择
        self.method_combo.currentIndexChanged.connect(self.toggle_parameters_visibility)  # 连接方法选择框的信号与槽函数
        self.main_layout.addWidget(self.method_combo)

        # 选择LPP方法
        self.lpp_method_label = QLabel("请选择邻域选择方法:")
        self.main_layout.addWidget(self.lpp_method_label)
        self.lpp_method_combo = QComboBox()
        self.lpp_method_combo.addItem("knn")
        self.lpp_method_combo.addItem("epsilon")
        self.lpp_method_combo.addItem("knn_epsilon")
        self.lpp_method_combo.addItem("adaptive_epsilon")
        self.lpp_method_combo.setCurrentText("knn")  # 设置初始值为当前选择
        self.lpp_method_combo.currentIndexChanged.connect(self.toggle_parameters_visibility)  # 连接方法选择框的信号与槽函数
        self.main_layout.addWidget(self.lpp_method_combo)

        self.k_label = QLabel("请输入数据点最近邻数量k:")
        self.main_layout.addWidget(self.k_label)
        self.k_input = QLineEdit()
        self.k_input.setText("5")  # 默认值为5
        self.main_layout.addWidget(self.k_input)

        self.t_label = QLabel("请输入热核参数t:")
        self.main_layout.addWidget(self.t_label)
        self.t_input = QLineEdit()
        self.t_input.setText("100000")  # 默认值为100000
        self.main_layout.addWidget(self.t_input)

        # 输入运行次数
        self.runs_label = QLabel("请输入运行次数:")
        self.main_layout.addWidget(self.runs_label)
        self.runs_input = QLineEdit()
        self.runs_input.setText("1")  # 默认值为1
        self.main_layout.addWidget(self.runs_input)

        # 执行程序按钮
        self.execute_button = QPushButton("执行程序")
        self.execute_button.clicked.connect(self.execute_algorithm)
        self.main_layout.addWidget(self.execute_button)

        # 信息显示框
        self.info_label = QLabel("程序信息显示:")
        self.main_layout.addWidget(self.info_label)

        self.info_textedit = QTextEdit()  # 用于显示函数信息的文本编辑框
        self.info_textedit.setReadOnly(True)  # 设置为只读模式
        self.main_layout.addWidget(self.info_textedit)

        self.eigenfaces_label = QLabel("最后一次运行的特征图像显示:")
        self.main_layout.addWidget(self.eigenfaces_label)

        self.canvas = FigureCanvas(plt.figure())
        self.main_layout.addWidget(self.canvas)

        # 初始化数据集路径变量
        self.dataset_path = self.default_dataset_path

    def center_on_screen(self):
        # 获取屏幕尺寸和窗口尺寸
        screen = QDesktopWidget().screenGeometry()
        window_size = self.geometry()

        # 计算窗口在屏幕中央的位置
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2

        # 移动窗口到屏幕中央
        self.move(x, y)

    def select_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.dataset_path = QFileDialog.getExistingDirectory(self, "选择数据集文件夹", options=options)
        if self.dataset_path:
            self.dataset_path_label.setText(f"数据集路径: {self.dataset_path}")
        if "ORL" in self.dataset_path:
            self.train_test_split_label.setVisible(True)
            self.train_test_split_combo.setVisible(True)
            self.target_size_label.setVisible(True)
            self.target_size_combo.setVisible(True)
            self.mnist_split_label.setVisible(False)
            self.mnist_split_combo.setVisible(False)
        elif "MNIST_ORG" in self.dataset_path:
            self.train_test_split_label.setVisible(False)
            self.train_test_split_combo.setVisible(False)
            self.target_size_label.setVisible(False)
            self.target_size_combo.setVisible(False)
            self.mnist_split_label.setVisible(True)
            self.mnist_split_combo.setVisible(True)
        elif "Reduced " in self.dataset_path:
            self.train_test_split_label.setVisible(False)
            self.train_test_split_combo.setVisible(False)
            self.target_size_label.setVisible(True)
            self.target_size_combo.setVisible(True)
            self.mnist_split_label.setVisible(False)
            self.mnist_split_combo.setVisible(False)

    def execute_algorithm(self):
        try:
            # 获取用户输入的参数值
            d = int(self.d_input.text())
            k = int(self.k_input.text())
            t = int(self.t_input.text())
            method = self.method_combo.currentText()
            lpp_method = self.lpp_method_combo.currentText()

            #如果读取的是ORL数据集，即self.dataset_path中含有"ORL"字符串
            if "ORL" in self.dataset_path:
                data_temp, labels_temp, faceshape_temp = read_images(self.dataset_path, target_size=None)
                train_test_split_ratio = float(self.train_test_split_combo.currentText())
            #如果读取的是MNIST数据集，即self.dataset_path中含有"MNIST"字符串
            elif "MNIST_ORG" in self.dataset_path:
                fraction = float(self.mnist_split_combo.currentText())
                train_data, train_labels, test_data, test_labels, faceshape = read_mnist_dataset(self.dataset_path, fraction=fraction)
                faceshape_temp = faceshape
            #如果读取的是mini_mnist数据集，即self.dataset_path中含有"Reduced"字符串
            elif "Reduced " in self.dataset_path:
                train_data, train_labels, test_data, test_labels, faceshape = read_mini_minst_images(self.dataset_path, target_size=None)
                faceshape_temp = faceshape

            # 获取目标尺寸并按选择缩放
            target_size_str = self.target_size_combo.currentText()
            if target_size_str == "100%":
                target_size = None
            else:
                percentage = int(target_size_str[:-1]) / 100.0
                target_size = (int(faceshape_temp[0] * percentage), int(faceshape_temp[1] * percentage))  # 使用 faceshape 确定原始尺寸
        
            runs = int(self.runs_input.text())  # 获取运行次数
            accuracies = []  # 用于存储每次运行的准确率
            start_time = time.time()  # 记录开始时间
            # 更改按钮文本为“程序执行中”
            self.execute_button.setText("程序执行中，请勿操作！")
            QApplication.processEvents()  # 强制刷新界面，立即显示按钮文本变更

            for _ in range(runs):
                if "ORL" in self.dataset_path:
                    data, labels, faceshape = read_images(self.dataset_path, target_size=target_size)
                    train_data, train_labels, test_data, test_labels = train_test_split(data, labels, train_test_split_ratio=train_test_split_ratio)
                elif "Reduced " in self.dataset_path:
                    train_data, train_labels, test_data, test_labels, faceshape = read_mini_minst_images(self.dataset_path, target_size=target_size)
                rate = 0.0  # 初始化识别率
                if method == "DLPP":
                    # 调用 DLPP 函数并接收返回的中间变量信息
                    F, L, B, objective_value, eigenfaces = DLPP(train_data, train_labels, d, lpp_method, k, t)
                    weight_matrix = np.dot(eigenfaces.T, train_data.T)

                    # 将最后一次运行的信息显示在文本编辑框中
                    if _ == runs - 1:
                        self.info_textedit.clear()
                        self.show_info("训练数据集形状:", train_data.T.shape)
                        self.show_info("类平均图像形状:", F.shape)
                        self.show_info("拉普拉斯矩阵形状:", L.shape)
                        self.show_info("类权重矩阵形状:", B.shape)
                        self.show_info("目标函数形状:", objective_value.shape)
                        self.show_info("特征图像形状:", eigenfaces.shape)
                        self.show_info("权重矩阵形状:", weight_matrix.shape)
                    # 识别率统计
                    wrong_times = 0
                    right_times = 0
                    for i in range(test_data.shape[0]):
                        flag = test_image(i, train_labels, test_labels, test_data[i], eigenfaces, weight_matrix)
                        if flag:
                            right_times += 1
                        else:
                            wrong_times += 1
                    rate = right_times / test_data.shape[0]

                elif method == "LPP":
                    # 调用 LPP 函数并接收返回的中间变量信息
                    train_data = train_data.T
                    eigenfaces = LPP(train_data, d, lpp_method, k, t)
                    weight_matrix = eigenfaces.T @ train_data
                    # 将最后一次运行的信息显示在文本编辑框中
                    if _ == runs - 1:
                        # 将信息显示在文本编辑框中
                        self.info_textedit.clear()
                        self.show_info("训练数据集形状:", train_data.shape)
                        self.show_info("特征图像形状:", eigenfaces.shape)
                        self.show_info("权重矩阵形状:", weight_matrix.shape)
                    # 识别率统计
                    wrong_times = 0
                    right_times = 0
                    for i in range(test_data.shape[0]):
                        flag = test_image(i, train_labels, test_labels, test_data[i], eigenfaces, weight_matrix)
                        if flag:
                            right_times += 1
                        else:
                            wrong_times += 1
                    rate = right_times / test_data.shape[0]

                elif method == "MLDA":
                    # 调用 MLDA 函数并接收返回的中间变量信息
                    eigenfaces, overall_mean, classes_means, Z, Sb, Sw, W_value = MLDA(train_data, train_labels, faceshape, d)
                    # 将最后一次运行的信息显示在文本编辑框中
                    if _ == runs - 1:
                        # 将信息显示在文本编辑框中
                        self.info_textedit.clear()
                        self.show_info("训练数据集形状:", train_data.T.shape)
                        self.show_info("平均图像形状:", overall_mean.shape)
                        self.show_info("类均值形状:", classes_means.shape)
                        self.show_info("Z形状:", Z.shape)
                        self.show_info("类间散度矩阵形状:", Sb.shape)
                        self.show_info("类内散度矩阵形状:", Sw.shape)
                        self.show_info("投影矩阵形状:", W_value.shape)
                        self.show_info("特征图像形状:", eigenfaces.shape)
                    # 识别率统计
                    wrong_times = 0
                    right_times = 0
                    for i in range(test_data.shape[0]):
                        flag = test_query_class_sample(eigenfaces, test_data[i], i, overall_mean, train_data, train_labels, test_labels)
                        if flag:
                                    right_times += 1         
                        else:
                            wrong_times += 1
                    rate = right_times / test_data.shape[0]

                elif method == "PCA":
                    # 调用 PCA 函数并接收返回的中间变量信息
                    eigenfaces, overall_mean = PCA(train_data, d)
                    weight_matrix = eigenfaces.T @ train_data.T
                    # 将最后一次运行的信息显示在文本编辑框中
                    if _ == runs - 1:
                        # 将信息显示在文本编辑框中
                        self.info_textedit.clear()
                        self.show_info("训练数据集形状:", train_data.T.shape)
                        self.show_info("平均图像形状:", overall_mean.shape)
                        self.show_info("特征脸形状:", eigenfaces.shape)
                        self.show_info("权重矩阵形状:", weight_matrix.shape)
                    # 识别率统计
                    wrong_times = 0
                    right_times = 0
                    for i in range(test_data.shape[0]):
                        flag = test_image(i, train_labels, test_labels, test_data[i], eigenfaces, weight_matrix)
                        if flag:
                            right_times += 1
                        else:
                            wrong_times += 1
                    rate = right_times / test_data.shape[0]

                accuracies.append(rate)  # 记录准确率

            average_accuracy = np.mean(accuracies)  # 计算平均准确率

            end_time = time.time()  # 记录结束时间
            execution_time = end_time - start_time  # 计算执行时间

            # 更新信息显示
            self.info_label.setText(f"平均图像识别正确率: {average_accuracy}\n程序运行时间: {execution_time:.2f} 秒")

            # 显示特征脸图像
            self.show_eigenfaces(eigenfaces, faceshape)

        except Exception as e:
            # 弹出错误窗口显示报错原因
            error_message = f"错误类型: {type(e).__name__}\n错误信息: {str(e)}"
            QMessageBox.critical(self, "错误", error_message)

        finally:
            # 执行完毕后还原按钮文本为“执行程序”
            self.execute_button.setText("执行程序")


    def show_info(self, title, info):
        # 将信息追加显示在文本编辑框中
        self.info_textedit.append(f"{title}: {info}")

    def show_eigenfaces(self, eigenfaces, faceshape):
        # 清空之前的图像
        self.canvas.figure.clear()

        # 计算特征脸的行数和列数以填充整个窗口
        num_faces = eigenfaces.shape[1]
        num_cols = int(np.ceil(np.sqrt(num_faces)))
        num_rows = int(np.ceil(num_faces / num_cols))

        # 显示特征脸并取消坐标轴
        for i in range(num_faces):
            ax = self.canvas.figure.add_subplot(num_rows, num_cols, i + 1)
            ax.imshow(eigenfaces[:, i].reshape(faceshape), cmap="gray")
            ax.axis('off')

        # 刷新画布
        self.canvas.draw()

    def toggle_parameters_visibility(self):
        # 获取当前选择的方法
        selected_lpp_method = self.lpp_method_combo.currentText()
        selected_method = self.method_combo.currentText()
        # 设置参数框的可见性
        if selected_method == "MLDA":
            self.k_label.setVisible(False)
            self.k_input.setVisible(False)
            self.t_label.setVisible(False)
            self.t_input.setVisible(False)
            self.d_label.setVisible(True)
            self.d_input.setVisible(True)
            self.lpp_method_label.setVisible(False)
            self.lpp_method_combo.setVisible(False)
        elif selected_method == "LPP": 
            if selected_lpp_method == "adaptive_epsilon":
                self.k_label.setVisible(False)
                self.k_input.setVisible(False)
            else:
                self.k_label.setVisible(True)
                self.k_input.setVisible(True)
            self.t_label.setVisible(True)
            self.t_input.setVisible(True)
            self.d_label.setVisible(True)
            self.d_input.setVisible(True)
            self.lpp_method_label.setVisible(True)
            self.lpp_method_combo.setVisible(True)
        elif selected_method == "DLPP":
            if selected_lpp_method == "adaptive_epsilon":
                self.k_label.setVisible(False)
                self.k_input.setVisible(False)
            else:
                self.k_label.setVisible(True)
                self.k_input.setVisible(True)
            self.t_label.setVisible(True)
            self.t_input.setVisible(True)
            self.d_label.setVisible(True)
            self.d_input.setVisible(True)
            self.lpp_method_label.setVisible(True)
            self.lpp_method_combo.setVisible(True)
        elif selected_method == "PCA":
            self.k_label.setVisible(False)
            self.k_input.setVisible(False)
            self.t_label.setVisible(False)
            self.t_input.setVisible(False)
            self.d_label.setVisible(True)
            self.d_input.setVisible(True)
            self.lpp_method_label.setVisible(False)
            self.lpp_method_combo.setVisible(False)
        else:
            raise ValueError(f"未知方法: {selected_method}")

if __name__ == "__main__":
    app = QApplication(argv)
    window = Window()
    window.show()
    exit(app.exec_())
