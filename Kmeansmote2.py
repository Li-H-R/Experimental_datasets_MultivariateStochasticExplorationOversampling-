from sklearn.cluster import KMeans
import numpy as np
import Markov_smote as ms
from sklearn.metrics.pairwise import euclidean_distances
import time

class KmeanSmoteTrainingMarchine():
    def __init__(self, input, labels):
        self.K_maxcluster = 9  # 最大聚类数
        self.input = input
        self.labels = labels
        self.threshold_IR = 1  # 过滤阈值为1
        self.density_power = input.shape[1]
        self.minority = 1
        self.majority = -1

    @staticmethod
    def one_hot(w, h, arr):
        z = np.zeros([w, h])  # 四行七列

        for i in range(w):  # 4
            j = int(arr)  # 拿到数组里面的数字
            z[i][j] = 1
        return z

    def kmeanself(self, inputdata):
        inputdata = self.input
        # 肘部法确定K值
        SSE = []
        for k in range(1, self.K_maxcluster):
            estimator = KMeans(n_clusters=k)
            estimator.fit(inputdata)
            _, co = np.unique(estimator.labels_, return_counts=True)
            # 保证每一簇的数量大于1个
            if 1 in co:
                break
            SSE.append(estimator.inertia_)

        # plt.plot(range(1, 9), SSE, 'o-')
        # plt.show()
        SSE = np.array(SSE)

        if len(SSE) > 2:
            good_SSE = SSE[SSE <= np.mean(SSE)][0]

            k_nvalue = np.argwhere(SSE == good_SSE)
            kmeans = KMeans(n_clusters=int(k_nvalue), random_state=4).fit(self.input)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
        else:
            kmeans = KMeans(n_clusters=len(SSE), random_state=4).fit(self.input)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

        return labels, centers

    def filtercluster(self, cluster_label, selected_labels):
        _, counts = np.unique(cluster_label, return_counts=True)
        filter_cluster_label = []  # 返回需要进行重采样的聚类标签
        # # ir_list = np.empty(0)
        # mino_list = np.empty(0)
        mio_num = self.input[selected_labels == self.minority].shape[0]
        maj_num = self.input[selected_labels == self.majority].shape[0]
        lamda = mio_num / maj_num

        IR_list = []
        moIR_list = []
        for i in range(len(counts)):
            cluster = selected_labels[cluster_label == i]
            majo = len(cluster[cluster == self.majority])
            mino = len(cluster[cluster == self.minority])
            IR = (mino) / (majo + 1e-8)
            IR_list.append(IR)
            moIR_list.append(mino / mio_num)

        IR_list = IR_list / np.sum(np.vstack(IR_list))
        ir_mean = np.vstack(IR_list) * lamda + (1 - lamda) * np.vstack(moIR_list)

        # atha = 15
        # beta = 15
        # self.threshold_IR = self.threshold_IR + (np.min(ir_mean) - self.threshold_IR) * beta * np.exp(-atha *
        #                                                                                  np.sqrt(np.var(ir_mean)))
        self.threshold_IR = 0.2
        # if max(ir_mean) > 0.5:
        #     self.threshold_IR = 0.5

        for j, ir in enumerate(ir_mean):
            if ir >= self.threshold_IR:
                filter_cluster_label.append(j)

        return filter_cluster_label

    def minority_QuantityProportion(self, cluster_assignment, cluster_label, mask_label):
        """
        :param cluster_assignment: 输入数据标签
        :param cluster_label:聚类标签
        :return:
        """
        minority_cluters = []
        minority_y = []
        mino_list = np.empty(0)
        for i in np.unique(cluster_assignment):
            cluster = self.input[cluster_label == i]
            mask = mask_label[cluster_label == i]
            minority_count = cluster[mask == self.minority].shape[0]
            mino_list = np.append(mino_list, minority_count)
            minority_cluters.append(cluster[mask == self.minority])  # 存储需要smote的数据
            minority_y.append(mask[mask == self.minority])
        if len(mino_list) != 1:
            sampling_weights = (1 - mino_list / np.sum(mino_list)) / (len(mino_list) - 1)
        else:
            sampling_weights = mino_list / np.sum(mino_list)

        return sampling_weights, minority_cluters, minority_y, mino_list

    def minority_density(self, cluster_assignment, cluster_label, mask_label):
        sparsity_factors = []
        minority_cluters = []
        minority_y = []
        mino_list = np.empty(0)
        for i in np.unique(cluster_assignment):
            cluster = self.input[cluster_label == i]
            mask = mask_label[cluster_label == i]
            minority_count = cluster[mask == self.minority].shape[0]
            mino_list = np.append(mino_list, minority_count)
            minority_cluters.append(cluster[mask == self.minority])  # 存储需要smote的数据
            minority_y.append(mask[mask == self.minority])
            distances = euclidean_distances(cluster[mask == self.minority])
            non_diagonal_distances = distances[
                ~np.eye(distances.shape[0], dtype=np.bool_)
            ]
            average_minority_distance = np.mean(non_diagonal_distances)
            if average_minority_distance == 0: average_minority_distance = 1e-1  # to avoid division by 0
            density_factor = minority_count / (average_minority_distance ** self.density_power)
            sparsity_factors.append(1 / density_factor)

        sparsity_factors = np.array(sparsity_factors)
        sparsity_sum = sparsity_factors.sum()
        if sparsity_sum == 0:
            sparsity_sum = 1  # to avoid division by zero

        sparsity_sum = np.full(sparsity_factors.shape, sparsity_sum, np.asarray(sparsity_sum).dtype)
        sampling_weights = (sparsity_factors / sparsity_sum)
        return sampling_weights, minority_cluters, minority_y, mino_list

    def kmeansmote_generate_data(self, select_label, sampling_num):
        """

        :param
        :param select_label: 选择需要生产的数据类--输入数据标签
        :param sampling_num: 生成样本数
        :return: 训练数据
        """
        # 确定输出的onehot向量

        label_all, counts = np.unique(self.labels, return_counts=True)
        output_onehot = self.one_hot(sampling_num, len(counts), select_label)

        # 将未选中的所有数据修改标签为 -1，选择的标签改为1
        new_labels = self.labels.copy()
        new_labels[new_labels != select_label] = -1
        new_labels[new_labels == select_label] = 1

        cluster_labels, _ = self.kmeanself(inputdata=self.input)
        # 滤波（滤出需要重采样的聚类群）
        filter_cluster = self.filtercluster(cluster_label=cluster_labels,
                                            selected_labels=new_labels)

        # import matplotlib.pyplot as plt
        # plt.scatter(self.input[cluster_labels == 2][:, 1], self.input[cluster_labels == 2][:, 2], label='minority class', c='blue', marker='*')
        # plt.scatter(self.input[cluster_labels == 0][:, 1], self.input[cluster_labels == 0][:, 2], label='minority class', c='blue', marker='.')
        # plt.scatter(self.input[cluster_labels == 3][:, 1], self.input[cluster_labels == 3][:, 2], label='minority class', c='red', marker='.')
        # # plt.scatter(self.input[cluster_labels == 1][:, 1], self.input[cluster_labels == 1][:, 2], label='minority class', c='black', marker='.')
        # plt.show()

        # # Step2.0: For each filtered cluster, compute the sampling weight based on its minority density.
        # weights, x, y, mino_num = self.minority_density(cluster_assignment=filter_cluster,
        #                                                 cluster_label=cluster_labels,
        #                                                 mask_label=new_labels)

        # Step2.1: For each filtered cluster, compute the sampling weight based on its minority density.
        weights, x, y, mino_num = self.minority_QuantityProportion(cluster_assignment=filter_cluster,
                                                                   cluster_label=cluster_labels,
                                                                   mask_label=new_labels)
        # 记录程序开始时间
        start_time = time.time()
        # 构造训练集
        sample = []
        sample_label = []
        sampling_remain_num = sampling_num - np.sum(mino_num)
        over_sampler = ms.Markov_SMMO()  # 方法实例化
        for i, weight in enumerate(weights):

            num = int(weight * sampling_remain_num + mino_num[i])
            # num = int(weight * sampling_num)

            if i == len(weights) - 1 and (i != 0):
                num = sampling_num - (np.vstack(sample)).shape[0]
            if x[i].shape[0] >= 2:

                # x_train, y_train = over_sampler.train_sample(x[i], y[i], num)  # 产生训练样本

                x_train, y_train = over_sampler.train_sample_Mcsmote(x[i], y[i], num, int(mino_num[i] / 2))  # 产生训练样本
                sample.append(x_train)
                sample_label.append(y_train)
            else:
                for _ in range(num):
                    direct_vector = np.zeros(x[i].shape)
                    x2 = np.append(direct_vector, x[i])
                    epsilon = np.random.random_sample()
                    sample.append(np.append(x2, epsilon))
                    sample_label.append(x[i])

        # final_input = np.hstack([np.vstack(sample), output_onehot])

        final_input = np.vstack(sample)
        final_label = np.vstack(sample_label)
        # 记录程序结束时间
        end_time = time.time()
        # 计算运行时间
        run_time = end_time - start_time
        print(f"程序运行时间：{run_time} 秒")


        import matplotlib.pyplot as plt
        dim_data = final_label.shape[1]
        a, b = 0, 1

        plt.scatter(self.input[:, a], self.input[:, b], label='minority class', c='black')
        plt.scatter(final_label[:, a], final_label[:, b], label='minority class', c='blue', marker='*')
        plt.scatter(final_input[:, a + dim_data], final_input[:, b + dim_data], label='minority class', c='red',
                    marker='*')

        plt.show()
        # print(np.vstack(sample_label).shape)
        return final_input, final_label

# s = KmeanSmoteTrainingMarchine(input=X, labels=y)
# c,v = s.kmeansmote_generate_data(select_label=0, sampling_num=200)
