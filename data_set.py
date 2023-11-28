import pandas as pd
import numpy as np
import spacy.parts_of_speech
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from imblance_dataset.db_read import ScadaRead_WP
from sklearn.model_selection import train_test_split
import imbalanced_databases as imbd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def dat_to_df(file, elements='auto'):
    data_lines = []
    with open(file, 'r') as file:
        for line in file:
            if line.strip() == '@data':
                break
        for line in file:
            data_lines.append(line.strip().split(','))  # 使用逗号作为字段分隔符
    # 为DataFrame的列命名
    if elements == 'auto':
        df1 = pd.DataFrame(data_lines).to_numpy()
    else:
        df1 = pd.DataFrame(data_lines, columns=elements).to_numpy()

    return df1


def label_to_my(yy):
    """
    将标签转换为0,1,2,3,....
    """
    uni = np.unique(yy)
    for i, u in enumerate(uni):
        yy[yy == u] = i + 100
    for k in range(len(uni)):
        yy[yy == k + 100] = k
    return yy.astype(np.int64)


def label_to_my_double(yy1, yy2):
    """
    将标签转换为0,1,2,3,....
    """
    uni = np.unique(np.concatenate([yy1, yy2]))
    for i, u in enumerate(uni):
        yy1[yy1 == u] = i + 100
        yy2[yy2 == u] = i + 100
    for k in range(len(uni)):
        yy1[yy1 == k + 100] = k
        yy2[yy2 == k + 100] = k
    return yy1.astype(np.int64), yy2.astype(np.int64)


class ImbalanceMultipleClassDataset:

    def load_toy(self):
        """
         人工数据集
        """
        random_seed = 42
        np.random.seed(random_seed)  # Set a seed for reproducibility
        A1_mean = [1, 1]
        A1_cov = [[2.0, 0.99], [1, 1]]
        A1 = np.random.multivariate_normal(A1_mean, A1_cov, 600)  # 依据指定的均值和协方差生成数据，size=5

        A2_mean = [1, 6]
        A2_cov = [[1, 0], [0, 1]]
        A2 = np.random.multivariate_normal(A2_mean, A2_cov, 30)  # 依据指定的均值和协方差生成数据，size=5

        A3_mean = [-3.5, 4]
        A3_cov = [[1, 0], [0, 1]]
        A3 = np.random.multivariate_normal(A3_mean, A3_cov, 30)  # 依据指定的均值和协方差生成数据，size=5

        A4_mean = [1, 1]
        A4_cov = [[2.0, 0.99], [1, 1]]
        A4 = np.random.multivariate_normal(A4_mean, A4_cov, 10)  # 依据指定的均值和协方差生成数据，size=5

        A4_mean1 = [-2, 0.1]
        A4_cov1 = [[0.5, 0], [0, 0.5]]
        A41 = np.random.multivariate_normal(A4_mean1, A4_cov1, 5)  # 依据指定的均值和协方差生成数据，size=5

        A5_mean = [2, -4]
        A5_cov = [[1, 0], [0, 1]]
        A5 = np.random.multivariate_normal(A5_mean, A5_cov, 30)  # 依据指定的均值和协方差生成数据，size=5

        moi_class1 = np.vstack([A2, A4])
        moi_class2 = np.vstack([A3, A41])
        moi_class3 = A5
        maj_class0 = A1

        y_maj = np.repeat(0, len(maj_class0))
        y_moi_class1 = np.repeat(1, len(moi_class1))
        y_moi_class2 = np.repeat(2, len(moi_class2))
        y_moi_class3 = np.repeat(3, len(moi_class3))

        y = np.hstack([y_maj, y_moi_class1, y_moi_class2, y_moi_class3])
        X = np.vstack([maj_class0, moi_class1, moi_class2, moi_class3])

        return {'data': X, 'target': y.astype(np.int64)}

    def load_toy_k_fold(self):

        x_data = self.load_toy()['data']
        y = self.load_toy()['target']
        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_autos(self):
        """
            Automobile Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Automobile data set without missing values, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		159
            5: Features.		25
            6: Classes.			6
            7: Missing values.	No
            8: IR: 				16.00

            9: Header.

            @relation unknow
            @attribute normalized-losses real [65.0, 256.0]
            @attribute make {alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo}
            @attribute fuel-type {diesel, gas}
            @attribute aspiration {std, turbo}
            @attribute num-of-doors {four, two}
            @attribute body-style {hardtop, wagon, sedan, hatchback, convertible}
            @attribute drive-wheels {4wd, fwd, rwd}
            @attribute engine-location {front, rear}
            @attribute wheel-base real [86.6, 115.6]
            @attribute length real [141.1, 202.6]
            @attribute width real [60.3, 71.7]
            @attribute height real [49.4, 59.8]
            @attribute curb-weight real [1488.0, 4066.0]
            @attribute engine-type {dohc, dohcv, l, ohc, ohcf, ohcv, rotor}
            @attribute num-of-cylinders {eight, five, four, six, three, twelve, two}
            @attribute engine-size real [61.0, 258.0]
            @attribute fuel-system {1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi}
            @attribute bore real [2.54, 3.94]
            @attribute stroke real [2.07, 4.17]
            @attribute compression-ratio real [7.0, 23.0]
            @attribute horsepower real [48.0, 200.0]
            @attribute peak-rpm real [4150.0, 6600.0]
            @attribute city-mpg real [15.0, 49.0]
            @attribute highway-mpg real [18.0, 54.0]
            @attribute price real [5118.0, 35056.0]
            @attribute symboling {-2, -1, 0, 1, 2, 3}
            @inputs normalized-losses, make, fuel-type, aspiration, num-of-doors, body-style, drive-wheels, engine-location, wheel-base, length, width, height, curb-weight, engine-type, num-of-cylinders, engine-size, fuel-system, bore, stroke, compression-ratio, horsepower, peak-rpm, city-mpg, highway-mpg, price
            @outputs symboling
        """

        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\autos\autos\autos.dat'
        # 获取属性名列表
        info = "normalized-losses, make, fuel-type, aspiration, num-of-doors, body-style, drive-wheels, engine-location, wheel-base, length, width, height, curb-weight, engine-type, num-of-cylinders, engine-size, fuel-system, bore, stroke, compression-ratio, horsepower, peak-rpm, city-mpg, highway-mpg, price"
        elements = info.split(', ')
        elements.append('symboling')

        df = dat_to_df(file, elements)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [1, 2, 3, 4, 5, 6, 7, 13, 14, 16, -1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_autos_k_fold(self):
        # 获取属性名列表
        info = "normalized-losses, make, fuel-type, aspiration, num-of-doors, body-style, drive-wheels, engine-location, wheel-base, length, width, height, curb-weight, engine-type, num-of-cylinders, engine-size, fuel-system, bore, stroke, compression-ratio, horsepower, peak-rpm, city-mpg, highway-mpg, price"
        elements = info.split(', ')
        elements.append('symboling')

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\autos\autos-5-fold\autos-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_autos()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1, elements=elements)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2, elements=elements)

            y_tra, y_tst = df_tra[:, -1].astype(float), df_tst[:, -1].astype(float)
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [1, 2, 3, 4, 5, 6, 7, 13, 14, 16, -1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [1, 2, 3, 4, 5, 6, 7, 13, 14, 16, -1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_balance(self):
        """
            An imbalanced version of the Balance Scale data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		625
            5: Features.		4
            6: Classes.			3
            7: Missing values.	No
            8: IR: 				5.88

            9: Header.

            @relation unknow
            @attribute left-weight real [1.0, 5.0]
            @attribute left-distance real [1.0, 5.0]
            @attribute right-weight real [1.0, 5.0]
            @attribute right-distance real [1.0, 5.0]
            @attribute class {L, B, R}
            @inputs left-weight, left-distance, right-weight, right-distance
            @outputs class

        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\balance\balance'
        file += r'\balance.dat'
        # 获取属性名列表
        info = "left-weight, left-distance, right-weight, right-distance"
        elements = info.split(', ')
        elements.append('class')

        df = dat_to_df(file, elements)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_balance_k_fold(self):
        # 获取属性名列表
        info = "left-weight, left-distance, right-weight, right-distance"
        elements = info.split(', ')
        elements.append('class')

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\balance\balance-5-fold\balance-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_balance()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1, elements)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2, elements)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_contraceptive(self):
        """
            Contraceptive Method Choice Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Contraceptive Method Choice data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		1473
            5: Features.		9
            6: Classes.			3
            7: Missing values.	No
            8: IR: 				1.89

            9: Header.

            @relation unknow
            @attribute a1 real [16.0, 49.0]
            @attribute a2 real [1.0, 4.0]
            @attribute a3 real [1.0, 4.0]
            @attribute a4 real [0.0, 16.0]
            @attribute a5 {0, 1}
            @attribute a6 {0, 1}
            @attribute a7 real [1.0, 4.0]
            @attribute a8 real [1.0, 4.0]
            @attribute a9 {0, 1}
            @attribute class {1, 2, 3}
            @inputs a1, a2, a3, a4, a5, a6, a7, a8, a9
            @outputs class

        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\contraceptive\contraceptive'
        file += r'\contraceptive.dat'
        # 获取属性名列表
        info = "a1, a2, a3, a4, a5, a6, a7, a8, a9"
        elements = info.split(', ')
        elements.append('class')

        df = dat_to_df(file, elements)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_contraceptive_k_fold(self):
        # 获取属性名列表
        info = "a1, a2, a3, a4, a5, a6, a7, a8, a9"
        elements = info.split(', ')
        elements.append('class')

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\contraceptive\contraceptive-5-fold\contraceptive-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_contraceptive()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1, elements)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2, elements)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_dermatology(self):
        """
            Dermatology Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Dermatology data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		366
            5: Features.		33
            6: Classes.			6
            7: Missing values.	Yes
            8: IR: 				5.55

            9: Header.

            @relation dermatology
            @attribute a1 integer[0,3]
            @attribute a2 integer[0,3]
            @attribute a3 integer[0,3]
            @attribute a4 integer[0,3]
            @attribute a5 integer[0,3]
            @attribute a6 integer[0,3]
            @attribute a7 integer[0,3]
            @attribute a8 integer[0,3]
            @attribute a9 integer[0,3]
            @attribute a10 integer[0,3]
            @attribute a11 integer[0,1]
            @attribute a12 integer[0,3]
            @attribute a13 integer[0,2]
            @attribute a14 integer[0,3]
            @attribute a15 integer[0,3]
            @attribute a16 integer[0,3]
            @attribute a17 integer[0,3]
            @attribute a18 integer[0,3]
            @attribute a19 integer[0,3]
            @attribute a20 integer[0,3]
            @attribute a21 integer[0,3]
            @attribute a22 integer[0,3]
            @attribute a23 integer[0,3]
            @attribute a24 integer[0,3]
            @attribute a25 integer[0,3]
            @attribute a26 integer[0,3]
            @attribute a27 integer[0,3]
            @attribute a28 integer[0,3]
            @attribute a29 integer[0,3]
            @attribute a30 integer[0,3]
            @attribute a31 integer[0,3]
            @attribute a32 integer[0,3]
            @attribute a33 integer[0,3]
            @attribute a34 integer[0,75]
            @attribute class integer[1,6]

        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\dermatology\dermatology'
        file += r'\dermatology.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_dermatology_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\dermatology\dermatology-5-fold\dermatology-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_dermatology()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_ecoli(self):
        """

            Ecoli Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Ecoli data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		336
            5: Features.		7
            6: Classes.			8
            7: Missing values.	No
            8: IR: 				71.50

            9: Header.

            @relation Ecoli-Bal
            @attribute mcg real [0.0, 0.89]
            @attribute gvh real [0.16, 1.0]
            @attribute lip real [0.48, 1.0]
            @attribute chg real [0.5, 1.0]
            @attribute aac real [0.0, 0.88]
            @attribute alm1 real [0.03, 1.0]
            @attribute alm2 real [0.0, 0.99]
            @attribute class {cp, im, pp, imU, om, omL, imL, imS}
            @inputs mcg, gvh, lip, chg, aac, alm1, alm2
            @outputs class


        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\ecoli\ecoli'
        file += r'\ecoli.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_ecoli_k_fold(self):

        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        X, y = self.load_ecoli()['data'], self.load_ecoli()['target']
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_glass(self):
        """

            Glass Identification Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Glass Identification data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		214
            5: Features.		9
            6: Classes.			7
            7: Missing values.	No
            8: IR: 				8.44

            9: Header.

            @relation Glass_Identification_Database
            @attribute RI real [1.51115, 1.53393]
            @attribute Na real [10.73, 17.38]
            @attribute Mg real [0.0, 4.49]
            @attribute Al real [0.29, 3.5]
            @attribute Si real [69.81, 75.41]
            @attribute K real [0.0, 6.21]
            @attribute Ca real [5.43, 16.19]
            @attribute Ba real [0.0, 3.15]
            @attribute Fe real [0.0, 0.51]
            @attribute typeGlass {1, 2, 3, 4, 5, 6, 7}
            @inputs RI, Na, Mg, Al, Si, K, Ca, Ba, Fe
            @outputs typeGlass


        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\glass\glass'
        file += r'\glass.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_glass_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\glass\glass-5-fold\glass-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_glass()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_hayes_roth(self):
        """

            Hayes-Roth Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Hayes-Roth data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Laboratory
            4: Instances.		132
            5: Features.		4
            6: Classes.			3
            7: Missing values.	No
            8: IR: 				1.70

            9: Header.

            @relation Hayes-Roth database
            @attribute hobby integer [1, 3]
            @attribute age integer [1, 4]
            @attribute educationalLevel integer [1, 4]
            @attribute maritalStatus integer [1, 4]
            @attribute class {1,2,3}
            @inputs hobby, age, educationalLevel, maritalStatus
            @outputs class



        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\hayes-roth\hayes-roth'
        file += r'\hayes-roth.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_hayes_roth_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\hayes-roth\hayes-roth-5-fold\hayes-roth-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_hayes_roth()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_new_thyroid(self):
        """

             Thyroid Disease (New Thyroid) Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Thyroid Disease (New Thyroid) data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		215
            5: Features.		5
            6: Classes.				3
            7: Missing values.	No
            8: IR: 				4.84

            9: Header.

            @relation New-Thyroid-Bal
            @attribute T3resin integer [65, 144]
            @attribute thyroxin real [0.5, 25.3]
            @attribute triiodothyronine real [0.2, 10.0]
            @attribute thyroidstimulating real [0.1, 56.4]
            @attribute TSH_value real [-0.7, 56.3]
            @attribute class {normal, hyper, hypo}
            @inputs T3resin, thyroxin, triiodothyronine, thyroidstimulating, TSH_value
            @outputs class




        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\new-thyroid\new-thyroid'
        file += r'\new-thyroid.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_new_thyroid_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\new-thyroid\new-thyroid-5-fold\new-thyroid-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_new_thyroid()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_pageblocks(self):
        """

            Page Blocks Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Page Blocks data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		548
            5: Features.		10
            6: Classes.			5
            7: Missing values.	No
            8: IR: 				164.00

            9: Header.

            @relation unknow
            @attribute height real [1.0, 804.0]
            @attribute lenght real [1.0, 553.0]
            @attribute area real [7.0, 143993.0]
            @attribute eccen real [0.0070, 537.0]
            @attribute p_black real [0.052, 1.0]
            @attribute p_and real [0.062, 1.0]
            @attribute mean_tr real [1.0, 4955.0]
            @attribute blackpix real [1.0, 33017.0]
            @attribute blackand real [7.0, 46133.0]
            @attribute wb_trans real [1.0, 3212.0]
            @attribute class {1, 2, 4, 5, 3}
            @inputs height, lenght, area, eccen, p_black, p_and, mean_tr, blackpix, blackand, wb_trans
            @outputs class
        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\pageblocks\pageblocks'
        file += r'\pageblocks.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_pageblocks_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\pageblocks\pageblocks-5-fold\pageblocks-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_pageblocks()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_penbased(self):
        """

            Pen-Based Recognition of Handwritten Digits Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Pen-Based Recognition of Handwritten Digits data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		1100
            5: Features.		16
            6: Classes.			10
            7: Missing values.	No
            8: IR: 				1.95

            9: Header.

            @relation unknow
            @attribute at1 real [0.0, 100.0]
            @attribute at2 real [0.0, 100.0]
            @attribute at3 real [0.0, 100.0]
            @attribute at4 real [0.0, 100.0]
            @attribute at5 real [0.0, 100.0]
            @attribute at6 real [0.0, 100.0]
            @attribute at7 real [0.0, 100.0]
            @attribute at8 real [0.0, 100.0]
            @attribute at9 real [0.0, 100.0]
            @attribute at10 real [0.0, 100.0]
            @attribute at11 real [0.0, 100.0]
            @attribute at12 real [0.0, 100.0]
            @attribute at13 real [0.0, 100.0]
            @attribute at14 real [0.0, 100.0]
            @attribute at15 real [0.0, 100.0]
            @attribute at16 real [0.0, 100.0]
            @attribute class {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            @inputs at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12, at13, at14, at15, at16
            @outputs class
        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\penbased\penbased'
        file += r'\penbased.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_penbased_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\penbased\penbased-5-fold\penbased-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_penbased()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_shuttle(self):
        """

            Statlog (Shuttle) Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Statlog (Shuttle) data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		2175
            5: Features.		9
            6: Classes.				7
            7: Missing values.	No
            8: IR: 				853.00

            9: Header.

            @relation unknow
            @attribute a1 integer [27, 126]
            @attribute a2 integer [-4821, 5075]
            @attribute a3 integer [21, 149]
            @attribute a4 integer [-3939, 3830]
            @attribute a5 integer [-188, 436]
            @attribute a6 integer [-13839, 13148]
            @attribute a7 integer [-48, 105]
            @attribute a8 integer [-353, 270]
            @attribute a9 integer [-356, 266]
            @attribute class {1,2,3,4,5,6,7}
            @inputs a1, a2, a3, a4, a5, a6, a7, a8, a9
            @outputs class

        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\shuttle\shuttle'
        file += r'\shuttle.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_shuttle_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\shuttle\shuttle-5-fold\shuttle-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_shuttle()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_thyroid(self):
        """

            Thyroid Disease (thyroid0387) Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Thyroid Disease (thyroid0387) data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		720
            5: Features.		21
            6: Classes.			3
            7: Missing values.	No
            8: IR: 				36.94

            9: Header.

            @relation thyroid
            @attribute Sintoma1 real [0.01, 0.97]
            @attribute Sintoma2 integer [0, 1]
            @attribute Sintoma3 integer [0, 1]
            @attribute Sintoma4 integer [0, 1]
            @attribute Sintoma5 integer [0, 1]
            @attribute Sintoma6 integer [0, 1]
            @attribute Sintoma7 integer [0, 1]
            @attribute Sintoma8 integer [0, 1]
            @attribute Sintoma9 integer [0, 1]
            @attribute Sintoma10 integer [0, 1]
            @attribute Sintoma11 integer [0, 1]
            @attribute Sintoma12 integer [0, 1]
            @attribute Sintoma13 integer [0, 1]
            @attribute Sintoma14 integer [0, 1]
            @attribute Sintoma15 integer [0, 1]
            @attribute Sintoma16 integer [0, 1]
            @attribute Sintoma17 real [0.0, 0.53]
            @attribute Sintoma18 real [0.0005, 0.18]
            @attribute Sintoma19 real [0.0020, 0.6]
            @attribute Sintoma20 real [0.017, 0.233]
            @attribute Sintoma21 real [0.0020, 0.642]
            @attribute class {1,2,3}
            @inputs Sintoma1, Sintoma2, Sintoma3, Sintoma4, Sintoma5, Sintoma6, Sintoma7, Sintoma8, Sintoma9, Sintoma10, Sintoma11, Sintoma12, Sintoma13, Sintoma14, Sintoma15, Sintoma16, Sintoma17, Sintoma18, Sintoma19, Sintoma20, Sintoma21
            @outputs class

        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\thyroid\thyroid'
        file += r'\thyroid.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_thyroid_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\thyroid\thyroid-5-fold\thyroid-5-'
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_thyroid()['data'])
        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)
            x_tra = np.delete(df_tra, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_wine(self):
        """

           Wine Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Wine data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		178
            5: Features.		13
            6: Classes.			3
            7: Missing values.	No
            8: IR: 				1.5

            9: Header.

            @relation wine
            @attribute at1 real [11.0, 14.9]
            @attribute at2 real [0.7, 5.8]
            @attribute at3 real [1.3, 3.3]
            @attribute at4 real [10.6, 30.0]
            @attribute at5 real [70.0, 162.0]
            @attribute at6 real [0.9, 3.9]
            @attribute at7 real [0.3, 5.1]
            @attribute at8 real [0.1, 0.7]
            @attribute at9 real [0.4, 3.6]
            @attribute at10 real [1.2, 13.0]
            @attribute at11 real [0.4, 1.8]
            @attribute at12 real [1.2, 4.0]
            @attribute at13 real [278.0, 1680.0]
            @attribute class {1, 2, 3}
            @inputs at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12, at13
            @outputs class


        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\wine\wine'
        file += r'\wine.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_wine_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\wine\wine-5-fold\wine-5-'

        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_wine()['data'])

        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_yeast(self):
        """
            Yeast Multi-class Imbalanced data set

            1: Description.

            An imbalanced version of the Yeast data set, where there are some classes with a small number of examples while other classes have a large number of examples.

            2: Type.			Imbalanced
            3: Origin.			Real world
            4: Instances.		1484
            5: Features.		8
            6: Classes.			10
            7: Missing values.	No
            8: IR: 				23.15

            9: Header.

            @relation yeastB
            @attribute mcg real [0.11, 1.0]
            @attribute gvh real [0.13, 1.0]
            @attribute alm real [0.21, 1.0]
            @attribute mit real [0.0, 1.0]
            @attribute erl real [0.5, 1.0]
            @attribute pox real [0.0, 0.83]
            @attribute vac real [0.0, 0.73]
            @attribute nuc real [0.0, 1.0]
            @attribute class {MIT,NUC,CYT,ME1,ME2,ME3,EXC,VAC,POX,ERL}
            @inputs mcg, gvh, alm, mit, erl, pox, vac, nuc
            @outputs class



        """
        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\yeast\yeast'
        file += r'\yeast.dat'

        df = dat_to_df(file)
        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [-1], axis=1).astype(float)
        return {'data': x_data, 'target': y_data}

    def load_yeast_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\yeast\yeast-5-fold\yeast-5-'

        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_yeast()['data'])

        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [-1], axis=1).astype(float)
            x_tst = np.delete(df_tst, [-1], axis=1).astype(float)
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_lymphography(self):

        file = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类\lymphography\lymphography'
        file += r'\lymphography.dat'

        df = dat_to_df(file)

        y_data = label_to_my(df[:, -1])
        x_data = np.delete(df, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18], axis=1).astype(float)

        return {'data': x_data, 'target': y_data}

    def load_lymphography_k_fold(self):

        file_str = r'C:\Users\admin\Desktop\新论文\第一篇文章\数据集\不平衡多分类'
        file_str += r'\lymphography\lymphography-5-fold\lymphography-5-'

        # 归一化
        scaler = MinMaxScaler()
        scaler.fit_transform(self.load_lymphography()['data'])

        keys, values = [], []
        # 读取数据文件并提取数据部分
        for i in range(5):
            file1 = file_str + str(i + 1) + 'tra.dat'
            df_tra = dat_to_df(file1)

            file2 = file_str + str(i + 1) + 'tst.dat'
            df_tst = dat_to_df(file2)

            y_tra, y_tst = df_tra[:, -1], df_tst[:, -1]
            y_tra, y_tst = label_to_my_double(y_tra, y_tst)

            x_tra = np.delete(df_tra, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18], axis=1).astype(float)
            x_tst = np.delete(df_tst, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18], axis=1).astype(float)

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_tra), 'target': y_tra})
            values.append({'data': scaler.transform(x_tst), 'target': y_tst})
        my_dict = {key: value for key, value in zip(keys, values)}
        return my_dict

    def load_toy_10v4(self):
        # 生成4类数据，增加cluster_std参数和高斯噪声
        x_data, y_data = make_classification(n_samples=400, n_features=10, n_classes=4, n_clusters_per_class=1,
                                             weights=[0.7, 0.1, 0.1, 0.2], n_informative=10, n_redundant=0,
                                             class_sep=1.4,
                                             random_state=42)

        return {'data': x_data, 'target': y_data}

    def load_toy_10v4_k_fold(self):
        # 生成4类数据，增加cluster_std参数和高斯噪声
        x_data, y = make_classification(n_samples=400, n_features=10, n_classes=4, n_clusters_per_class=1,
                                        weights=[0.7, 0.1, 0.1, 0.2], n_informative=10, n_redundant=0, class_sep=1.4,
                                        random_state=42)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_toy_moon(self):
        np.random.seed(42)
        # datasets.make_moons?
        my_datas = datasets.make_moons(n_samples=2000,
                                       noise=0.3)

        X, y = my_datas

        X2 = 0.7 * X[y == 0] + [2, 0.5]
        X3 = 0.7 * X[y == 0] + [0, 0.5]
        X4 = X[y == 1]

        id_X2 = np.random.choice(range(len(X2)), 100)
        X2 = X2[id_X2]

        id_X3 = np.random.choice(range(len(X3)), 100)
        X3 = X3[id_X3]

        id_X4 = np.random.choice(range(len(X4)), 500)
        X4 = X4[id_X4]

        y_maj = np.repeat(0, len(X4))
        y_moi_class1 = np.repeat(1, len(X2))
        y_moi_class2 = np.repeat(2, len(X3))

        y_data = np.hstack([y_maj, y_moi_class1, y_moi_class2])
        x_data = np.vstack([X4, X2, X3])

        return {'data': x_data, 'target': y_data}

    def load_toy_moon_k_fold(self):

        x_data, y = self.load_toy_moon()['data'], self.load_toy_moon()['target']
        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_toy_circles(self):
        np.random.seed(42)
        # datasets.make_circles？
        my_datas = datasets.make_circles(n_samples=2000,
                                         noise=0.3,
                                         factor=0.1, shuffle=True)

        my_datas1 = datasets.make_circles(n_samples=500,
                                          noise=0.3,
                                          factor=0.2, )

        X, y = my_datas

        X3 = X[y == 1][:1000]
        X1, y1 = my_datas1
        X2 = X1[y1 == 0] * 2
        X4 = X[y == 0][:250]

        y_maj = np.repeat(0, len(X3))
        y_moi_class1 = np.repeat(1, len(X2))
        y_moi_class2 = np.repeat(2, len(X4))

        y_data = np.hstack([y_maj, y_moi_class1, y_moi_class2])
        x_data = np.vstack([X3, X2, X4])

        return {'data': x_data, 'target': y_data}

    def load_toy_circles_k_fold(self):
        x_data, y = self.load_toy_circles()['data'], self.load_toy_circles()['target']
        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def fault_get(self, wt_name, a, b):
        data_name = wt_name
        query0 = f"SELECT 有功功率," \
                 f"30秒平均风速," \
                 f"风向角, 风轮转速, " \
                 f"桨距角1, 桨距角2, 桨距角3, 发电机转速, " \
                 f"驱动端轴承温度, 高速轴承温度, 低速轴承温度, 齿轮箱入口油温, " \
                 f"齿轮箱冷却水温度 FROM {data_name}"
        wp_get = ScadaRead_WP()
        data2, _ = wp_get.scada_data(query0, data_name)
        return data2[a:b, ]

    def load_WindTurbine(self):
        """
        属性：有功功率, 无功功率, 电网A相电压, " \
                 f"电网B相电压, 电网C相电压, 电网A相电流, 电网B相电流, " \
                 f"电网C相电流, 电网频率, 功率因子, 30秒平均风速," \
                 f"风向角, 风轮转速, " \
                 f"桨距角1, 桨距角2, 桨距角3, 发电机转速, " \
                 f"发电机定子U温度, 发电机定子V温度, 发电机定子W温度, " \
                 f"冷却风扇进口温度, 冷却风扇出口温度, 自由端轴承温度, " \
                 f"驱动端轴承温度, 高速轴承温度, 低速轴承温度, 齿轮箱入口油温, " \
                 f"齿轮箱冷却水温度

            故障类[正常0， 变桨系统1， 齿轮系统2， 发电机系统3]
        """
        fault_pitch_dataset = self.fault_get("b13_201807", 17432, 17605)

        fault_gearbox_dataset = self.fault_get("b11_201711", 29633, 30318)
        fault_generator_dataset = self.fault_get("b12_201807", 42425, 43100)
        fault_converter_dataset = self.fault_get("b12_201803", 36715, 36831)

        # 正常
        normal_dataset = self.fault_get("b11_201703", 34867, 42731)

        y_maj = np.repeat(0, len(normal_dataset))
        y_moi_class1 = np.repeat(1, len(fault_pitch_dataset))
        y_moi_class2 = np.repeat(2, len(fault_gearbox_dataset))
        y_moi_class3 = np.repeat(3, len(fault_generator_dataset))
        y_moi_class4 = np.repeat(4, len(fault_converter_dataset))

        y_data = np.hstack([y_maj, y_moi_class1, y_moi_class2, y_moi_class3, y_moi_class4])
        features = np.concatenate(
            [normal_dataset, fault_pitch_dataset, fault_gearbox_dataset, fault_generator_dataset,
             fault_converter_dataset], axis=0)

        return {'data': features, 'target': y_data}

    def load_WindTurbine_k_fold(self):
        x_data, y = self.load_WindTurbine()['data'], self.load_WindTurbine()['target']

        scaler = MinMaxScaler()

        np.random.seed(42)
        scaler.fit_transform(x_data)
        # Split the data into training and testing sets (70% training, 30% testing)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.5, random_state=42, stratify=y)
        # Introduce noise to the training set
        np.random.seed(42)  # For reproducibility
        noise_indices = np.where(y_train == 0)
        # noise_indices = np.random.choice(noise_indices[0], int(0.7 * len(y_train[y_train == 0])))
        # y_train[noise_indices] = np.random.choice([1, 2, 3, 4], len(noise_indices))

        # import matplotlib.pyplot as plt
        # from sklearn.manifold import TSNE
        # # Apply t-SNE
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_result = tsne.fit_transform(x_train)
        #
        # # Create a scatter plot for t-SNE visualization
        # plt.figure(figsize=(10, 8))
        # plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_train, alpha=0.5, edgecolors='w',
        #             s=100)
        #
        # # Add labels, title, or any other customization if needed
        # plt.title('t-SNE Visualization')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        #
        # # Show the plot
        # plt.show()

        keys, values = [], []
        for i in range(1):
            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': scaler.transform(x_train), 'target': y_train})
            values.append({'data': scaler.transform(x_test), 'target': y_test})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_abalone(self):
        abalone = imbd.load_abalone9_18()
        x_data, y_data = abalone['data'], abalone['target']
        x_data = np.delete(x_data, [0], axis=1).astype(float)

        return {'data': x_data, 'target': y_data.astype(np.int64)}

    def load_binary_abalone_k_fold(self):
        x_data, y = self.load_binary_abalone()['data'], self.load_binary_abalone()['target']
        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_breast(self):
        cancers = datasets.load_breast_cancer()
        x_data, y_data = cancers['data'], cancers['target']
        # x_data = np.delete(x_data, [0], axis=1).astype(float)

        return {'data': x_data, 'target': y_data.astype(np.int64)}

    def load_binary_breast_k_fold(self):
        x_data, y = self.load_binary_breast()['data'], self.load_binary_breast()['target']

        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_glass(self):
        glass = imbd.load_glass1()
        x_data, y_data = glass['data'], glass['target']
        # x_data = np.delete(x_data, [0], axis=1).astype(float)

        return {'data': x_data, 'target': y_data.astype(np.int64)}

    def load_binary_glass_k_fold(self):
        x_data, y = self.load_binary_glass()['data'], self.load_binary_glass()['target']

        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_pima(self):
        pima = imbd.load_pima()
        x_data, y_data = pima['data'], pima['target']
        # x_data = np.delete(x_data, [0], axis=1).astype(float)

        return {'data': x_data, 'target': y_data.astype(np.int64)}

    def load_binary_pima_k_fold(self):
        x_data, y = self.load_binary_pima()['data'], self.load_binary_pima()['target']

        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_segment(self):
        pima = imbd.load_segment0()
        x_data, y_data = pima['data'], pima['target']
        x_data = np.delete(x_data, [2,3,4,5,6,7,8], axis=1).astype(float)

        return {'data': x_data, 'target': y_data.astype(np.int64)}

    def load_binary_segment_k_fold(self):
        x_data, y = self.load_binary_segment()['data'], self.load_binary_segment()['target']

        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_satimage(self):
        pima = imbd.load_satimage()
        x_data, y_data = pima['data'], pima['target']
        x_data = np.delete(x_data, [2,3,4,5,6,7,8], axis=1).astype(float)

        return {'data': x_data, 'target': y_data.astype(np.int64)}

    def load_binary_satimage_k_fold(self):
        x_data, y = self.load_binary_satimage()['data'], self.load_binary_satimage()['target']

        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_iris(self):
        pima = imbd.load_iris0()
        x_data, y_data = pima['data'], pima['target']

        return {'data': x_data, 'target': y_data.astype(np.int64)}

    def load_binary_iris_k_fold(self):
        x_data, y = self.load_binary_iris()['data'], self.load_binary_iris()['target']

        scaler = MinMaxScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_wt(self):
        x_data = np.load(r'F:\pycharm_project\SMOTE\imblance_dataset\WindTurbine_X.npy', allow_pickle=True)
        y_data = np.load(r'F:\pycharm_project\SMOTE\imblance_dataset\WindTurbine_Y.npy', allow_pickle=True)

        # plt.scatter(x_data[y_data==0][:, 0], x_data[y_data==0][:, 1])
        # plt.scatter(x_data[y_data == 1][:, 0], x_data[y_data == 1][:, 1])
        # plt.show()
        return {'data': x_data, 'target': y_data.astype(np.int64)}

    def load_binary_wt_k_fold(self):
        x_data, y = self.load_binary_wt()['data'], self.load_binary_wt()['target']

        scaler = StandardScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict

    def load_binary_wt_2_fold(self):
        x_data, y = self.load_binary_wt()['data'], self.load_binary_wt()['target']

        scaler = StandardScaler()
        X = scaler.fit_transform(x_data)
        # 创建一个基于类别的分层折叠交叉验证对象
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        # 利用交叉验证对象获取每一折的训练集和测试集索引
        i = 0
        keys, values = [], []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            keys.append('train' + str(i))
            keys.append('test' + str(i))
            values.append({'data': X_train_fold, 'target': y_train_fold})
            values.append({'data': X_test_fold, 'target': y_test_fold})
            i += 1
        my_dict = {key: value for key, value in zip(keys, values)}

        return my_dict


if __name__ == '__main__':
    imdataset = ImbalanceMultipleClassDataset()
    dd = imdataset.load_WindTurbine()

    uu, cc = np.unique(dd['target'], return_counts=True)
    aa=  np.mean(np.max(cc)/cc)
    print(dd['data'].shape)
    print(aa, len(dd['target']), cc)

