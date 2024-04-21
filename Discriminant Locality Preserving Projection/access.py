bigList = []
        # 导入人脸模型库
        faceCascade = cv2.CascadeClassifier(r'D:\Desktop\pca\haarcascade_frontalface_default.xml')
        # 遍历10个人
        if (data_choose == "yale"): i1 = 16;j1 = 12;la = 11;a3 = 100;b3 = 80
        if (data_choose == "feret"): i1 = 59;j1 = 8;la = 7;a3 = 80;b3 = 80
        if (data_choose == "orl"): i1 = 41;j1 = 11;la = 10;a3 = 112;b3 = 92
        for i in range(1, i1):
            # 遍历每个人的10张照片
            for j in range(1, j1):
                list = []
                # 直接读入灰度照片   orl1_1
                # yale 16 12
                if (data_choose == "yale"):
                    if (i < 10): image = cv2.imread(
                        "D:\\Desktop\\bishe\\Face\\face10080\\subject0" + str(i) + "_" + str(j) + ".bmp", 0)
                    if (i > 9): image = cv2.imread(
                        "D:\\Desktop\\bishe\\Face\\face10080\\subject" + str(i) + "_" + str(j) + ".bmp", 0)

                # feret 59 8
                if (data_choose == "feret"):
                    image = cv2.imread(
                        "D:\\Desktop\\bishe\\Face\\FERET_Face\\FERET-0" + str(i) + "\\0" + str(j) + ".tif", 0)
                # orl 41 11
                if (data_choose == "orl"):
                    image = cv2.imread("D:\\Desktop\\bishe\\Face\\ORL_Faces\\s" + str(i) + "\\" + str(j) + ".pgm",
                                       0)
                # image = cv2.imread("D:\\Desktop\\bishe\\Face\\ORL56_46\\orl" + str(i) + "_" + str(j) + ".bmp", 0)
                faces = faceCascade.detectMultiScale(image, 1.3, 5)
                # for (x, y, w, h) in faces:
                #     # 裁剪人脸区域为 128 * 128 大小
                #     cutResize = cv2.resize(image[y:y + h, x:x + w], (128, 128),
                #                            interpolation=cv2.INTER_CUBIC)
                # 遍历图片行数
                for x in range(image.shape[0]):
                    # 遍历图片每一行的每一列
                    for y in range(image.shape[1]):
                        # 将每一处的灰度值添加至列表
                        list.append(image[x, y])
                bigList.append(list)

        trainFaceMat1 = numpy.mat(bigList)  # 得到训练样本矩阵
        print(trainFaceMat1.shape)
        trainFaceMat = PCA(n_components=100).fit_transform(trainFaceMat1)

        m = np.zeros((trainFaceMat.shape[0], 1))
        for i in range(0, m.shape[0]):
            m[i][0] = int(i / la + 1)

        trainFaceMat = np.c_[m, trainFaceMat]

        # 假设你有一个包含列名的列表 column_names
        column_names = ['Class label'] + [f'Column_{i}' for i in range(trainFaceMat.shape[1] - 1)]

        # 创建 Pandas DataFrame，并设置第一列为自定义索引，其他列为默认索引
        df = pd.DataFrame(trainFaceMat, columns=column_names)
        # df.set_index('Class label', inplace=True)
        data = df
        # 输出带有索引的 DataFrame
        print(data)
        # 每一类数据包含的样本个数
        data['Class label'].value_counts()
        data.head()

        labels = data['Class label'].unique()
        num_labels = len(labels)

        # 数据集设置：X为样本特征数据，y为目标数据，即标注结果
        X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values

        # 数据集划分： 将数据集划分为训练集和测试集数据（测试集数据为30%，训练集为70%）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            stratify=y,
                                                            random_state=1)

        # 实例化
        sc = StandardScaler()

        # 对数据集进行标准化（一般情况下我们在训练集中进行均值和方差的计算，直接在测试集中使用）
        # X_train_std = sc.fit_transform(X_train)
        # X_test_std = sc.transform(X_test)
        X_train_std = X_train
        X_test_std = X_test