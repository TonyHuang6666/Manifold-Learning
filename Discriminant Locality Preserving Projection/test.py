import sys
import os
import time
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLineEdit, QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from DLPP import *


class DLPPWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DLPP程序图形界面")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # 默认数据集路径
        self.default_dataset_path = "D:\OneDrive - email.szu.edu.cn\Manifold Learning\Discriminant Locality Preserving Projection\ORL"

        # 选择数据集路径
        self.dataset_label = QLabel("选择数据集文件夹:")
        self.main_layout.addWidget(self.dataset_label)
        self.dataset_path_label = QLabel(f"ORL数据集路径: {self.default_dataset_path}")  # 显示默认数据集文件夹的路径
        self.main_layout.addWidget(self.dataset_path_label)
        self.dataset_button = QPushButton("选择其他数据集")
        self.dataset_button.clicked.connect(self.select_dataset)
        self.main_layout.addWidget(self.dataset_button)

        self.d_label = QLabel("请输入d:")
        self.main_layout.addWidget(self.d_label)
        self.d_input = QLineEdit()
        self.d_input.setText("70")  # 默认值为70
        self.main_layout.addWidget(self.d_input)

        self.k_label = QLabel("请输入k:")
        self.main_layout.addWidget(self.k_label)
        self.k_input = QLineEdit()
        self.k_input.setText("100")  # 默认值为100
        self.main_layout.addWidget(self.k_input)

        self.t_label = QLabel("请输入t:")
        self.main_layout.addWidget(self.t_label)
        self.t_input = QLineEdit()
        self.t_input.setText("60000")  # 默认值为60000
        self.main_layout.addWidget(self.t_input)

        self.lpp_label = QLabel("请选择lpp_method:")
        self.main_layout.addWidget(self.lpp_label)
        self.lpp_combo = QComboBox()
        self.lpp_combo.addItem("knn")
        self.lpp_combo.addItem("epsilon")
        self.main_layout.addWidget(self.lpp_combo)

        self.train_test_split_label = QLabel("请选择训练集/测试集划分比例:")
        self.main_layout.addWidget(self.train_test_split_label)
        self.train_test_split_combo = QComboBox()
        for ratio in range(5, 100, 5):
            ratio_decimal = ratio / 100.0
            self.train_test_split_combo.addItem("{:.2f}".format(ratio_decimal))
        self.train_test_split_combo.setCurrentText("0.50")  # 设置初始值为当前选择
        self.main_layout.addWidget(self.train_test_split_combo)

        self.info_label = QLabel("DLPP程序信息将在这里显示")
        self.main_layout.addWidget(self.info_label)

        self.canvas = FigureCanvas(plt.figure())
        self.main_layout.addWidget(self.canvas)

        self.execute_button = QPushButton("执行DLPP程序")
        self.execute_button.clicked.connect(self.execute_DLPP)
        self.main_layout.addWidget(self.execute_button)

        # 初始化数据集路径变量
        self.dataset_path = self.default_dataset_path

    def select_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.dataset_path = QFileDialog.getExistingDirectory(self, "选择数据集文件夹", options=options)
        if self.dataset_path:
            self.dataset_path_label.setText(f"数据集路径: {self.dataset_path}")

    def execute_DLPP(self):
        # 获取用户输入的参数值
        d = int(self.d_input.text())
        k = int(self.k_input.text())
        t = int(self.t_input.text())
        lpp_method = self.lpp_combo.currentText()
        train_test_split_ratio = float(self.train_test_split_combo.currentText())

        start_time = time.time()  # 记录开始时间

        # 调用 DLPP.py 文件中的相关函数，并获取其输出信息
        data, labels, faceshape = read_images(self.dataset_path)
        train_data, train_labels, test_data, test_labels = train_test_split(data, labels, train_test_split_ratio=train_test_split_ratio)

        dlpp_eigenfaces = DLPP(train_data, train_labels, d, lpp_method, k, t)
        overall_mean = np.mean(train_data, axis=0).reshape(-1, 1)
        dlpp_weight_matrix = np.dot(dlpp_eigenfaces.T, train_data.T- overall_mean)

        # 识别率统计
        wrong_times = 0
        right_times = 0
        for i in range(test_data.shape[0]):
            flag = test_image(i, faceshape, overall_mean, train_labels, train_data, test_labels, test_data[i], dlpp_eigenfaces, dlpp_weight_matrix)
            if flag:
                right_times += 1
            else:
                wrong_times += 1
        rate = right_times / (right_times + wrong_times)

        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间

        # 更新信息显示
        self.info_label.setText(f"Recognition Rate: {rate}\nExecution Time: {execution_time:.2f} seconds")

        # 显示特征脸图像
        self.show_eigenfaces(dlpp_eigenfaces, faceshape)

    def show_eigenfaces(self, eigenfaces, faceshape):
        # 清空之前的图像
        self.canvas.figure.clear()

        # 显示前16个特征脸
        num_faces = min(eigenfaces.shape[1], 16)
        num_rows = num_faces // 4
        num_cols = 4
        for i in range(num_faces):
            ax = self.canvas.figure.add_subplot(num_rows, num_cols, i + 1)
            ax.imshow(eigenfaces[:, i].reshape(faceshape), cmap="gray")

        # 刷新画布
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DLPPWindow()
    window.show()
    sys.exit(app.exec_())
